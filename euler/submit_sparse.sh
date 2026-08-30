#!/usr/bin/bash
# Submit the sparse-assembly harness (test_sparse_version.py) as ONE job.
#
#   bash euler/submit_sparse.sh                          # all three tests, both encodings
#   TESTS=equivalence SOLVE=1 bash euler/submit_sparse.sh
#   ENCODINGS=bigm FACTORS=1,2,4,8,16 bash euler/submit_sparse.sh
#   EXCLUSIVE=1 F=8 L=2 H=6 FACTORS=1,2,4,8 bash euler/submit_sparse.sh
#
# There is no array and no merge here, unlike submit.sh / submit_studies.sh. That
# is not an omission: the harness mostly BUILDS models and never solves in the
# scaling test, so the whole run is minutes rather than node-hours. Sharding
# would buy nothing and would let the two halves of a comparison land on nodes
# with different clock speeds, which is precisely what a build benchmark must
# not do.
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TESTS=${TESTS:-equivalence,scaling,solve}
ENCODINGS=${ENCODINGS:-indicator,bigm}
NAME=${NAME:-sparse}
OUT=${OUT:-$PROJECT/results}
# Measured, not guessed: the default payload is ~1 min of compute and peaks
# under 0.2 GB. 30 min of wall time backfills far better on Euler than 4 h, and
# a single core is right because the build -- which is what this harness
# measures -- is serial Python. Raise CPUS only if you set SOLVE=1 on a grid
# whose instances are actually hard.
WALL=${WALL:-00:30:00}
CPUS=${CPUS:-1}
MEM=${MEM:-4g}
# A build benchmark on a shared node measures the neighbours as much as the code.
# Set EXCLUSIVE=1 for any number you intend to put in a report.
EXCLUSIVE=${EXCLUSIVE:-0}
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d%H%M)}

cd "$PROJECT"                         # so SLURM_SUBMIT_DIR == project root
mkdir -p logs "$OUT"                  # Slurm will NOT create the log directory

for t in ${TESTS//,/ }; do
    case "$t" in
        equivalence|scaling|solve) ;;
        *) echo "ERROR: unknown test '$t'; pick from equivalence,scaling,solve" >&2
           exit 1 ;;
    esac
done
# ENCODINGS names the MILP ENCODING, not the formulation string: the harness
# turns each into its (loop, sparse) pair -- indicator/sparse and
# bigm/bigm_sparse -- because the assembly is what is under test, and a
# comparison across encodings would be comparing two different models.
for e in ${ENCODINGS//,/ }; do
    case "$e" in
        indicator|bigm) ;;
        *) echo "ERROR: unknown encoding '$e'; pick from indicator,bigm." >&2
           echo "       ('sparse' and 'bigm_sparse' are the sparse-assembly" >&2
           echo "        twins of those two, and the harness pairs them up" >&2
           echo "        itself -- you do not name them here.)" >&2
           exit 1 ;;
    esac
done

# Fail here, not after printing a submission plan we cannot carry out.
if [ ! -d "$PROJECT/.venv-euler" ]; then
    echo "ERROR: no .venv-euler in $PROJECT -- run 'bash euler/setup_euler.sh' first" >&2
    exit 1
fi
MISSING=""
for f in euler/run_sparse.sbatch "$PROJECT/test_sparse_version.py"; do
    [ -f "$f" ] || MISSING="$MISSING $f"
done
if [ -n "$MISSING" ]; then
    echo "ERROR: missing file(s):$MISSING" >&2
    echo "       test_sparse_version.py goes in the project ROOT, next to" >&2
    echo "       test.py. The .sbatch files travel with the .sh files -- commit" >&2
    echo "       and push the whole euler/ folder, then 'git pull' here (or scp" >&2
    echo "       them up)." >&2
    exit 1
fi

# rainflow_sparse is checked by IMPORT, not by path. The package may sit in
# src/fleet_management/ or in fleet_management/, and once `pip install -e` has
# run it can be resolved from somewhere else entirely -- so guessing a directory
# is both fragile and beside the point. What has to be true is that the
# interpreter the job will use can import it, which is what this asks.
source "$PROJECT/.venv-euler/bin/activate" 2>/dev/null || true
PKG_CHECK=$(python - <<'PKGPY' 2>&1
import sys
try:
    from fleet_management.degradation_model import rainflow_sparse
    from fleet_management.degradation_model.base import FORMULATIONS
except Exception as exc:
    print(f"FAIL {type(exc).__name__}: {exc}")
    sys.exit(0)
missing = [f for f in ("indicator", "bigm", "sparse", "bigm_sparse")
           if f not in FORMULATIONS]
if missing:
    print(f"FAIL base.FORMULATIONS is missing {missing}: base.py and "
          f"rainflow_v2.py have not been updated alongside rainflow_sparse.py")
else:
    print(f"OK {rainflow_sparse.__file__}")
PKGPY
)
case "$PKG_CHECK" in
    OK\ *)
        echo "package   : ${PKG_CHECK#OK }" ;;
    *)
        echo "ERROR: cannot import fleet_management.degradation_model.rainflow_sparse" >&2
        echo "       ${PKG_CHECK#FAIL }" >&2
        echo "       It goes in the package next to base.py and rainflow_v2.py --" >&2
        echo "       src/fleet_management/degradation_model/ in a src layout," >&2
        echo "       fleet_management/degradation_model/ otherwise. Locate the" >&2
        echo "       package with:" >&2
        echo "         python -c 'import fleet_management.degradation_model as m; print(m.__path__)'" >&2
        echo "       A stale build/ directory or a second copy on sys.path can" >&2
        echo "       also shadow the one you just edited." >&2
        exit 1 ;;
esac
# A stale harness is the cheapest failure to catch and the most annoying to
# diagnose from a log: argparse exits 2 with a usage dump.
for flag in --tests --encodings --factors --repeats --run-stamp --solve; do
    if ! grep -q -- "\"$flag\"" "$PROJECT/test_sparse_version.py"; then
        echo "ERROR: $PROJECT/test_sparse_version.py does not support $flag --" >&2
        echo "       it is an older version than these job scripts expect." >&2
        echo "       Update it and check:" >&2
        echo "         python test_sparse_version.py --help | grep -- '--encodings'" >&2
        exit 1
    fi
done

# CRLF in a batch script makes Slurm fail in confusing ways ("not found" for a
# file that exists, because the interpreter becomes /usr/bin/bash\r).
if grep -qU $'\r' euler/run_sparse.sbatch 2>/dev/null; then
    echo "ERROR: euler/run_sparse.sbatch has Windows (CRLF) line endings." >&2
    echo "       Fix with: sed -i 's/\r$//' euler/run_sparse.sbatch   (and add a" >&2
    echo "       .gitattributes rule '*.sbatch text eol=lf')" >&2
    exit 1
fi

echo "project   : $PROJECT"
echo "tests     : $TESTS"
echo "encodings : $ENCODINGS  (each compared loop vs sparse assembly)"
echo "base case : F=${F:-8} M=${M:-2} L=${L:-2} H=${H:-6}  ladder x${FACTORS:-1,2,4,8}"
echo "run folder: $OUT/${RUN_STAMP}_${NAME}"
if [ "$EXCLUSIVE" = "1" ]; then
    echo "node      : EXCLUSIVE (timings are quotable)"
else
    echo "node      : shared -- fine for (S1) correctness, but a build benchmark"
    echo "            on a shared node measures the neighbours too. Use"
    echo "            EXCLUSIVE=1 for numbers that go in a report."
fi
if git -C "$PROJECT" rev-parse --git-dir >/dev/null 2>&1; then
    BRANCH=$(git -C "$PROJECT" rev-parse --abbrev-ref HEAD)
    COMMIT=$(git -C "$PROJECT" rev-parse --short HEAD)
    echo "code      : $BRANCH @ $COMMIT"
    if [ -n "$(git -C "$PROJECT" status --porcelain)" ]; then
        echo "WARNING working tree is dirty -- the commit will not identify the"
        echo "        code that produced these results. Commit first."
    fi
fi
echo "note      : do not edit the code until the job has finished"

EXTRA_SBATCH=()
[ "$EXCLUSIVE" = "1" ] && EXTRA_SBATCH+=(--exclusive)

JOB_ID=$(PROJECT=$PROJECT TESTS=$TESTS ENCODINGS=$ENCODINGS NAME=$NAME OUT=$OUT \
    RUN_STAMP=$RUN_STAMP \
    F="${F:-8}" M="${M:-2}" L="${L:-2}" H="${H:-6}" \
    FACTORS="${FACTORS:-1,2,4,8}" REPEATS="${REPEATS:-3}" \
    SOLVE="${SOLVE:-0}" EXTRA="${EXTRA:-}" \
    sbatch --parsable --time="$WALL" --cpus-per-task="$CPUS" \
           --mem-per-cpu="$MEM" "${EXTRA_SBATCH[@]}" euler/run_sparse.sbatch)
echo "job       : $JOB_ID  ($CPUS cpus, $MEM/cpu, $WALL)"
echo "            (the build is serial Python; CPUS only affects Gurobi in the"
echo "             solve test and under SOLVE=1)"
echo
echo "watch with : squeue -u \$USER   /   myjobs -j $JOB_ID"
echo "results in : $OUT/${RUN_STAMP}_${NAME}/"
echo "              summary.txt        verdicts and the timing tables"
echo "              equivalence.csv    one row per (encoding, bound, impl, repair, repl)"
echo "              scaling.csv        one row per (encoding, parameter, value, assembly)"
echo "              sparse_build.png   build time and speed-up vs each parameter"
echo
echo "the job exits non-zero if (S1) finds a mismatch, so check:"
echo "  sacct -j $JOB_ID --format=JobID,State,ExitCode"
