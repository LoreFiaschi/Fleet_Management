#!/usr/bin/bash
# Submit a sharded test plus its merge, wired with a dependency.
#   bash euler/submit.sh sweep 20
#   TEST=impl EXTRA="--impls tangent,pwl,exact" bash euler/submit.sh impl 12
# The merge uses 'afterany' rather than 'afterok' on purpose: if one shard dies,
# you still get the merged results for every shard that finished, plus a warning
# in the summary telling you which runs are missing.
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST=${1:-sweep}
NSHARDS=${2:-20}
NAME=${NAME:-bound_tightness}
OUT=${OUT:-$PROJECT/results}
MAXPAR=${MAXPAR:-10}                  # concurrent array tasks

cd "$PROJECT"                         # so SLURM_SUBMIT_DIR == project root
mkdir -p logs "$OUT"                  # Slurm will NOT create the log directory

if [ ! -d "$PROJECT/.venv-euler" ]; then
    echo "ERROR: no .venv-euler in $PROJECT -- run 'bash euler/setup_euler.sh' first" >&2
    exit 1
fi

echo "project   : $PROJECT"
if git -C "$PROJECT" rev-parse --git-dir >/dev/null 2>&1; then
    BRANCH=$(git -C "$PROJECT" rev-parse --abbrev-ref HEAD)
    COMMIT=$(git -C "$PROJECT" rev-parse --short HEAD)
    echo "code      : $BRANCH @ $COMMIT"
    if [ -n "$(git -C "$PROJECT" status --porcelain)" ]; then
        echo "WARNING working tree is dirty -- the commit recorded in the results"
        echo "        will not identify the code that produced them. Commit first,"
        echo "        or accept the '+dirty' marker in the CSV."
    fi
fi
# Array tasks read test.py when each task STARTS, not when you submit. Editing the
# code or pulling while the array is queued means later tasks run different code
# than earlier ones; the merge step detects this and warns.
echo "note      : do not edit the code or 'git pull' until the array has finished"
ARRAY_ID=$(PROJECT=$PROJECT TEST=$TEST NAME=$NAME OUT=$OUT sbatch --parsable \
    --array=0-$((NSHARDS - 1))%"$MAXPAR" euler/run_array.sbatch)
echo "array job : $ARRAY_ID  ($NSHARDS shards, max $MAXPAR running at once)"

MERGE_ID=$(PROJECT=$PROJECT TEST=$TEST NAME=$NAME OUT=$OUT sbatch --parsable \
    --dependency=afterany:"$ARRAY_ID" euler/merge.sbatch)
echo "merge job : $MERGE_ID  (runs after the array finishes)"
echo
echo "watch with : squeue -u \$USER   /   myjobs -j $ARRAY_ID"
echo "results in : $OUT/*_${NAME}_${TEST}_merged/"
