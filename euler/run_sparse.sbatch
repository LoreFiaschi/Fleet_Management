#!/usr/bin/bash
#SBATCH --job-name=fleet_sparse
#SBATCH --ntasks=1                    # always 1 on Euler
#SBATCH --cpus-per-task=1             # nothing here is parallel; see the note below
#SBATCH --mem-per-cpu=4g              # measured peak is <0.2 GB at the default ladder
#SBATCH --time=00:30:00               # measured payload is ~1 min; short jobs backfill
#
# These three are deliberately modest. submit_sparse.sh overrides them from
# WALL / CPUS / MEM, and an over-large request is not free: "Requested node
# configuration is not available" is Slurm refusing a combination that no node
# it can offer you matches, and a 4 h / 8 GB-per-core ask is a good way to
# provoke it for a job that needs one minute and 200 MB.
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# test_sparse_version.py on Euler: is the sparse assembly the same program, and
# is it faster to build?
#
# NOT an array job, deliberately. The other harnesses shard because they solve
# hundreds of MIPs; this one mostly BUILDS models and never solves in the
# scaling test, so the whole thing is minutes, and sharding would only add a
# merge step and a way for the two halves of a comparison to land on nodes with
# different clock speeds. One task, one node, one folder.
#
#   bash euler/submit_sparse.sh                        # equivalence + scaling + solve
#   TESTS=scaling FACTORS=1,2,4,8 bash euler/submit_sparse.sh
#   ENCODINGS=bigm bash euler/submit_sparse.sh         # the combination that pays
set -euo pipefail

# Slurm copies the batch script to a spool dir, so $0 is useless for locating the
# project. SLURM_SUBMIT_DIR is the directory sbatch was called from --
# submit_sparse.sh always calls it from the project root.
PROJECT=${PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}

TESTS=${TESTS:-equivalence,scaling,solve}
# Which MILP ENCODINGS to test the assembly on. Both by default, and that is the
# point: 'indicator' saturates at ~2x because its state recursions are general
# indicator constraints with no matrix form, while 'bigm' is all linear rows and
# reaches 5-7x. One encoding alone gives a misleading picture of either.
ENCODINGS=${ENCODINGS:-indicator,bigm}
NAME=${NAME:-sparse}
OUT=${OUT:-$PROJECT/results}
RUN_STAMP=${RUN_STAMP:?RUN_STAMP must be exported by euler/submit_sparse.sh}
EXTRA=${EXTRA:-}

# Base case and ladder for the (S2) build-cost measurement. The defaults are
# sized so the top rung is a genuinely large model (F=64, L=16, T=48 is ~1M rows)
# without needing more than the 4 h wall time.
F=${F:-8}; M=${M:-2}; L=${L:-2}; H=${H:-6}
FACTORS=${FACTORS:-1,2,4,8}
REPEATS=${REPEATS:-3}
# (S1) also solves both models and requires an EXACT match of the objective AND
# the LP relaxation when SOLVE=1. Off by default: it multiplies the grid by two
# MIPs plus two LPs per case, and the structural comparison is the check with
# teeth anyway.
SOLVE=${SOLVE:-0}
# Seconds Gurobi may spend on ONE solve, in (S3) and under SOLVE=1. Distinct
# from the Slurm --time, which bounds the whole task. Only the solve tests use
# it: (S2) never solves, so it cannot be affected.
SOLVE_TL=${SOLVE_TL:-120}

# `module` is a shell function set up by the login profile. A batch script runs a
# NON-login shell, so on some setups it is undefined here and the task dies
# instantly with "module: command not found" -- which under `set -e` looks like a
# failure with no useful message. Bootstrap it if needed.
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
    echo "       GRB_LICENSE_FILE is unset and every model build fails (an" >&2
    echo "       environment is created even when nothing is solved). Aborting." >&2
    module list 2>&1 >&2
    exit 1
fi

if [ ! -f "$PROJECT/.venv-euler/bin/activate" ]; then
    echo "ERROR: no venv at $PROJECT/.venv-euler -- run euler/setup_euler.sh" >&2
    exit 1
fi
source "$PROJECT/.venv-euler/bin/activate"
cd "$PROJECT"

# Fail loudly here rather than inside the harness.
# NOTE `import gurobipy` succeeds WITHOUT a licence -- the licence is only checked
# when an environment is created, which this harness does for every build. So the
# preflight has to actually create one.
# NOTE scipy is REQUIRED here, unlike for run_studies.py where it only improves a
# metric: the sparse assembly is COO -> CSR before it reaches Gurobi, so without
# scipy every 'sparse' build raises ImportError and the whole run is empty.
set +e
python - <<'PREFLIGHT'
import numpy, gurobipy as gp
from scipy import sparse as sp
m = gp.Model()                     # <- this is what needs the licence
m.Params.OutputFlag = 0
x = m.addVar(ub=2)
m.setObjective(x, gp.GRB.MAXIMIZE)
m.optimize()
assert abs(m.ObjVal - 2.0) < 1e-9
# the exact call path the sparse assembly depends on: scipy CSR into addMConstr
m2 = gp.Model(); m2.Params.OutputFlag = 0
v = m2.addVars(3)
m2.update()
m2.addMConstr(sp.csr_matrix([[1.0, 1.0, 0.0]]), list(v.values()), '<',
              numpy.array([1.0]))
m2.optimize()
print(f"preflight ok: gurobipy {gp.gurobi.version()}, numpy {numpy.__version__}, "
      f"scipy {sp.__name__.split('.')[0]} present, addMConstr accepts scipy CSR")
PREFLIGHT
PF_RC=$?
set -e
if [ $PF_RC -ne 0 ]; then
    echo "ERROR: preflight failed (exit $PF_RC). Diagnostics:" >&2
    echo "  python           : $(which python)" >&2
    echo "  GRB_LICENSE_FILE : ${GRB_LICENSE_FILE:-<UNSET -- gurobi module not loaded>}" >&2
    echo "  scipy            : $(python -c 'import scipy;print(scipy.__version__)' 2>&1)" >&2
    module list 2>&1 >&2 || true
    gurobi_cl --tokens 2>&1 >&2 || echo "  gurobi_cl --tokens failed" >&2
    exit 1
fi

# Version check inside the task too, not only at submit time: a task can start
# minutes after submission, and argparse's failure mode is a usage dump that
# buries the real problem. Costs ~1 s.
for flag in --tests --encodings --factors --repeats --run-stamp --no-plots; do
    if ! python test_sparse_version.py --help 2>/dev/null | grep -q -- "$flag"; then
        echo "ERROR: this test_sparse_version.py does not support $flag -- it is" >&2
        echo "       older than these job scripts. On the login node: git pull" >&2
        echo "       (or scp the current file up), then verify with:" >&2
        echo "         python test_sparse_version.py --help | grep -c -- '--encodings'" >&2
        exit 1
    fi
done
# The four formulation values must exist in the package, or --encodings bigm
# fails deep inside a build instead of here.
python - <<'FORMCHECK' || exit 1
from fleet_management.degradation_model.base import FORMULATIONS, assembly_of
need = ("indicator", "bigm", "sparse", "bigm_sparse")
missing = [f for f in need if f not in FORMULATIONS]
if missing:
    raise SystemExit(
        f"ERROR: base.FORMULATIONS is missing {missing}. This package predates "
        f"the sparse assembly; put rainflow_sparse.py in the package next to "
        f"base.py (src/fleet_management/degradation_model/ in a src layout) "
        f"and update base.py / rainflow_v2.py.")
assert assembly_of("bigm_sparse") == "sparse"
print("formulations ok:", FORMULATIONS)
FORMCHECK

echo "project=$PROJECT"
echo "python=$(which python)  gurobi=$(python -c 'import gurobipy;print(gurobipy.gurobi.version())')"
echo "host=$(hostname)  cpus=${SLURM_CPUS_PER_TASK:-1}"
echo "tests=$TESTS  encodings=$ENCODINGS  stamp=$RUN_STAMP"
echo "base case: F=$F M=$M L=$L H=$H   ladder x$FACTORS   repeats=$REPEATS"
echo "solve=$SOLVE  per-solve time limit=${SOLVE_TL}s"
echo "started $(date)"

# NumPy/SciPy do not parallelise anything on this path (no BLAS calls -- it is
# ravel_multi_index, repeat, fancy indexing and a COO -> CSR conversion), but
# OpenBLAS still spins up a thread pool sized to the cores it can see when it is
# imported. Idle threads on a shared node are exactly what get_inefficient_jobs
# reports, so cap them at the allocation.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# --threads: this harness is SINGLE-THREADED where it matters. (S2) measures how
#   long it takes to BUILD a model in Python -- a serial loop of API calls plus a
#   COO -> CSR conversion, none of which Gurobi or NumPy parallelises -- and it
#   never solves at all. Threads therefore affect only (S3) and --solve, and it
#   is pinned to what Slurm actually granted rather than left to Gurobi's "one
#   thread per core I can SEE", which on a shared Euler node oversubscribes the
#   machine and shows up in get_inefficient_jobs.
#   Keep CPUS constant across submissions whose solve times you intend to
#   compare, for the same reason run_studies.py does.
# Reserve a whole node with --exclusive (see submit_sparse.sh) if you intend to
# quote the timing numbers: a build benchmark on a shared node measures the
# neighbours as much as the code -- and asking for one core does NOT stop a
# neighbour from saturating the memory bandwidth you are measuring against.
ARGS=(--tests "$TESTS"
      --encodings "$ENCODINGS"
      --name "$NAME" --out "$OUT"
      --run-stamp "$RUN_STAMP"
      --F "$F" --M "$M" --L "$L" --H "$H"
      --factors "$FACTORS"
      --repeats "$REPEATS"
      --threads "${SLURM_CPUS_PER_TASK:-1}"
      --time-limit "$SOLVE_TL")
[ "$SOLVE" = "1" ] && ARGS+=(--solve)

python test_sparse_version.py "${ARGS[@]}" $EXTRA
RC=$?

echo "finished $(date)"
# The harness exits 1 when (S1) found a mismatch. Propagate it: a green Slurm
# job for a run that proved the two builds differ is worse than useless.
exit $RC
