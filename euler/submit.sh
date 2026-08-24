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
# MILP encoding (rainflow_v2). FORM picks one encoding for the whole submission;
# TEST=formulation instead solves BOTH (FORMS) and checks they agree.
FORM=${FORM:-}
FORMS=${FORMS:-indicator,bigm}
BIGM=${BIGM:-}
for f in ${FORM:-} ${FORMS//,/ }; do
    case "$f" in
        indicator|bigm) ;;
        *) echo "ERROR: unknown formulation '$f'; pick from indicator,bigm" >&2
           exit 1 ;;
    esac
done
# A run folder is results/<stamp>_<test>/ and a shard writes results_shard<k>.csv
# into it. Two submissions that differ ONLY in the encoding would therefore
# overwrite each other whenever they share a stamp. Give each encoding its own
# results subtree by default (still overridable with OUT=).
if [ -n "$FORM" ]; then
    OUT=${OUT:-$PROJECT/results/$FORM}
else
    OUT=${OUT:-$PROJECT/results}
fi
MAXPAR=${MAXPAR:-10}                  # concurrent array tasks
# One timestamp for the WHOLE run. Every array task starts at a different minute,
# so if each computed its own the shards would land in different folders and could
# never be merged. Format YYYYMMDDHHMM -> results/<stamp>_<test>/
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d%H%M)}

cd "$PROJECT"                         # so SLURM_SUBMIT_DIR == project root
mkdir -p logs "$OUT"                  # Slurm will NOT create the log directory

# Fail here, not after printing a submission plan we cannot carry out.
if [ ! -d "$PROJECT/.venv-euler" ]; then
    echo "ERROR: no .venv-euler in $PROJECT -- run 'bash euler/setup_euler.sh' first" >&2
    exit 1
fi
MISSING=""
for f in euler/run_array.sbatch euler/merge.sbatch "$PROJECT/test.py"; do
    [ -f "$f" ] || MISSING="$MISSING $f"
done
if [ -n "$MISSING" ]; then
    echo "ERROR: missing file(s):$MISSING" >&2
    echo "       The .sbatch files travel with the .sh files -- commit and push the" >&2
    echo "       whole euler/ folder, then 'git pull' here (or scp them up)." >&2
    exit 1
fi
# A stale test.py is the single most wasteful failure here: every array task dies
# in seconds with argparse exit code 2 and you only find out from the logs. The
# batch scripts pass --threads and --shard, so refuse to submit if this copy of
# test.py does not understand them.
for flag in --threads --shard --merge --gurobi-params --run-stamp \
            --formulation --formulations; do
    if ! grep -q -- "\"$flag\"" "$PROJECT/test.py"; then
        echo "ERROR: $PROJECT/test.py does not support $flag -- it is an older" >&2
        echo "       version than these job scripts expect." >&2
        echo "       Update it (git pull, or scp the current test.py up) and check:" >&2
        echo "         python test.py --help | grep -E -- '--threads|--shard|--merge'" >&2
        exit 1
    fi
done

# CRLF in a batch script makes Slurm fail in confusing ways ("not found" for a
# file that exists, because the interpreter becomes /usr/bin/bash\r).
for f in euler/run_array.sbatch euler/merge.sbatch; do
    if grep -qU $'\r' "$f" 2>/dev/null; then
        echo "ERROR: $f has Windows (CRLF) line endings." >&2
        echo "       Fix with: sed -i 's/\r$//' $f   (and add a .gitattributes" >&2
        echo "       rule '*.sbatch text eol=lf' so it does not come back)" >&2
        exit 1
    fi
done

echo "project   : $PROJECT"
echo "run folder: $OUT/${RUN_STAMP}_${TEST}"
if [ "$TEST" = "formulation" ]; then
    echo "encoding  : $FORMS  (solved BOTH and compared -- (H4))"
    case "$FORMS" in
      *,*) ;;
      *) echo "ERROR: the 'formulation' test compares encodings, so FORMS needs" >&2
         echo "       both, e.g. FORMS=indicator,bigm" >&2; exit 1 ;;
    esac
else
    echo "encoding  : ${FORM:-indicator (harness default)}${BIGM:+  bigM=$BIGM}"
fi
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
case "${EXTRA:-}" in
  *--gurobi-params*)
    echo "note      : EXTRA sets --gurobi-params. That is now MERGED with the job"
    echo "            script's memory guard (NodefileStart/SoftMemLimit), but the"
    echo "            intended slot is GUROBI_PARAMS=... which reads more clearly."
    ;;
esac
# NSHARDS is exported so the job never has to infer the array size itself.
if [ "$TEST" = "case" ] && [ -z "${CASES:-}" ]; then
    echo "ERROR: 'case' needs CASES=name1,name2 naming input files, e.g." >&2
    echo "         CASES=easy,hard bash euler/submit.sh case 2" >&2
    exit 1
fi

ARRAY_ID=$(PROJECT=$PROJECT TEST=$TEST NAME=$NAME OUT=$OUT NSHARDS=$NSHARDS \
    RUN_STAMP=$RUN_STAMP CASES="${CASES:-}" INPUT_DIR="${INPUT_DIR:-input}" \
    FORM="$FORM" FORMS="$FORMS" BIGM="$BIGM" \
    sbatch --parsable --array=0-$((NSHARDS - 1))%"$MAXPAR" euler/run_array.sbatch)
echo "array job : $ARRAY_ID  ($NSHARDS shards, max $MAXPAR running at once)"

MERGE_ID=$(PROJECT=$PROJECT TEST=$TEST NAME=$NAME OUT=$OUT RUN_STAMP=$RUN_STAMP \
    CASES="${CASES:-}" INPUT_DIR="${INPUT_DIR:-input}" \
    sbatch --parsable --dependency=afterany:"$ARRAY_ID" euler/merge.sbatch)
echo "merge job : $MERGE_ID  (runs after the array finishes)"
echo
echo "watch with : squeue -u \$USER   /   myjobs -j $ARRAY_ID"
echo "results in : $OUT/${RUN_STAMP}_${TEST}/  (merged_summary.txt when done)"
