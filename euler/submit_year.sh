#!/usr/bin/bash
# Submit the VBZ year instance as one array task per solver strategy, plus its
# merge, wired with a dependency.
#
#   bash euler/submit_year.sh                                  # all 5 strategies
#   bash euler/submit_year.sh vbz_man12e_year_solve warm,warm-norel
#   SOLVE_TL=36000 WALL=12:00:00 CKPT=600 bash euler/submit_year.sh
#
# The merge uses 'afterany' rather than 'afterok' on purpose: if one strategy
# dies you still get the ranking for the ones that finished.
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CASE=${1:-vbz_man12e_year_solve}
STRATEGIES=${2:-cold,warm,warm-norel,warm-bound,warm-pwl}
NAME=${NAME:-vbz_case}
TEST=${TEST:-year}
OUT=${OUT:-$PROJECT/results}
INPUT_DIR=${INPUT_DIR:-input}
MAXPAR=${MAXPAR:-5}
WALL=${WALL:-24:00:00}
CPUS=${CPUS:-8}
MEM=${MEM:-4g}
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d%H%M)}

cd "$PROJECT"
mkdir -p logs "$OUT"

if [ ! -d "$PROJECT/.venv-euler" ]; then
    echo "ERROR: no .venv-euler in $PROJECT -- run 'bash euler/setup_euler.sh' first" >&2
    exit 1
fi
MISSING=""
for f in euler/run_year_array.sbatch euler/merge_year.sbatch \
         "$PROJECT/run_year.py" "$PROJECT/test.py" \
         "$PROJECT/$INPUT_DIR/$CASE.yaml"; do
    [ -f "$f" ] || MISSING="$MISSING $f"
done
if [ -n "$MISSING" ]; then
    echo "ERROR: missing file(s):$MISSING" >&2
    echo "       run_year.py goes in the project ROOT (next to test.py); the" >&2
    echo "       instance YAML goes in $INPUT_DIR/; the .sbatch files in euler/.\n       run_year.py imports test.py BY PATH for the run-folder layout\n       and the CSV schema, so BOTH must be in the project root." >&2
    exit 1
fi
for flag in --strategy --gurobi-params --run-stamp --threads --checkpoint-every; do
    if ! grep -q -- "\"$flag\"" "$PROJECT/run_year.py"; then
        echo "ERROR: $PROJECT/run_year.py does not support $flag -- older than" >&2
        echo "       these job scripts." >&2
        exit 1
    fi
done
# CRLF in a batch script makes Slurm fail confusingly ("not found" for a file
# that exists, because the interpreter becomes /usr/bin/bash\r).
for f in euler/run_year_array.sbatch euler/merge_year.sbatch; do
    if grep -qU $'\r' "$f" 2>/dev/null; then
        echo "ERROR: $f has Windows (CRLF) line endings." >&2
        echo "       Fix with: sed -i 's/\r$//' $f   (and add a .gitattributes" >&2
        echo "       rule '*.sbatch text eol=lf' so it does not come back)" >&2
        exit 1
    fi
done

NSTRAT=$(awk -F, '{print NF}' <<< "$STRATEGIES")
echo "project    : $PROJECT"
echo "case       : $INPUT_DIR/$CASE.yaml"
echo "strategies : $STRATEGIES  ($NSTRAT array tasks)"
echo "run folder : $OUT/${RUN_STAMP}_${TEST}"
if git -C "$PROJECT" rev-parse --git-dir >/dev/null 2>&1; then
    echo "code       : $(git -C "$PROJECT" rev-parse --abbrev-ref HEAD) @ $(git -C "$PROJECT" rev-parse --short HEAD)"
    if [ -n "$(git -C "$PROJECT" status --porcelain)" ]; then
        echo "WARNING working tree is dirty -- the commit recorded in the results"
        echo "        will not identify the code that produced them."
    fi
fi

# Cheap pre-flight that needs no licence: certify that a feasible schedule
# exists before spending 5 x 20 h finding out the solver cannot locate one.
if [ -f "$PROJECT/feas_oracle.py" ]; then
    echo
    echo "=== feasibility witness (feas_oracle.py) ========================="
    set +e
    source "$PROJECT/.venv-euler/bin/activate" 2>/dev/null
    python feas_oracle.py "$INPUT_DIR/$CASE.yaml" \
        | grep -E "max mu|tangent slack|capacity_k|witness objective|structural"
    set -e
    echo "=================================================================="
fi
echo "note       : do not edit the code or 'git pull' until the array finishes"

ARRAY_ID=$(PROJECT=$PROJECT CASE=$CASE STRATEGIES=$STRATEGIES NAME=$NAME OUT=$OUT \
    RUN_STAMP=$RUN_STAMP INPUT_DIR=$INPUT_DIR TEST=$TEST CKPT="${CKPT:-900}" \
    SOLVE_TL="${SOLVE_TL:-72000}" \
    MIP_GAP="${MIP_GAP:-0.05}" GUROBI_PARAMS="${GUROBI_PARAMS:-}" EXTRA="${EXTRA:-}" \
    sbatch --parsable --array=0-$((NSTRAT - 1))%"$MAXPAR" \
           --time="$WALL" --cpus-per-task="$CPUS" --mem-per-cpu="$MEM" \
           euler/run_year_array.sbatch)
echo "array job  : $ARRAY_ID  ($NSTRAT tasks, max $MAXPAR at once, $CPUS cpus, $WALL)"

MERGE_ID=$(PROJECT=$PROJECT TEST=$TEST OUT=$OUT RUN_STAMP=$RUN_STAMP \
    sbatch --parsable --dependency=afterany:"$ARRAY_ID" euler/merge_year.sbatch)
echo "merge job  : $MERGE_ID  (runs after the array finishes)"
echo
echo "watch with : squeue -u \$USER   /   myjobs -j $ARRAY_ID"
echo "results in : $OUT/${RUN_STAMP}_${TEST}/  (merged_summary.txt when done)"
