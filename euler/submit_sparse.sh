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
WALL=${WALL:-04:00:00}
CPUS=${CPUS:-2}
MEM=${MEM:-8g}
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
for f in euler/run_sparse.sbatch "$PROJECT/test_sparse_version.py" \
         "$PROJECT/fleet_management/degradation_model/rainflow_sparse.py"; do
    [ -f "$f" ] || MISSING="$MISSING $f"
done
if [ -n "$MISSING" ]; then
    echo "ERROR: missing file(s):$MISSING" >&2
    echo "       test_sparse_version.py goes in the project ROOT (next to" >&2
    echo "       test.py); rainflow_sparse.py goes in the package next to" >&2
    echo "       base.py and rainflow_v2.py. The .sbatch files travel with the" >&2
    echo "       .sh files -- commit and push the whole euler/ folder, then" >&2
    echo "       'git pull' here (or scp them up)." >&2
    exit 1
fi
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
