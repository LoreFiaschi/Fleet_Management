#!/usr/bin/bash
# Submit a sharded study run plus its merge, wired with a dependency.
#
#   bash euler/submit_studies.sh scaling 12
#   bash euler/submit_studies.sh scaling,horizon,heatmap 20
#   SOLVE_TL=1800 EXTRA="--trace on" bash euler/submit_studies.sh convergence 4
#
# The merge uses 'afterany' rather than 'afterok' on purpose: if one shard dies,
# you still get the merged results for every shard that finished, plus a warning
# in the summary telling you which runs are missing.
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STUDIES=${1:-scaling}
NSHARDS=${2:-12}
NAME=${NAME:-studies}
# MILP encoding x assembly (rainflow_v2 / rainflow_sparse).
#   indicator | bigm        the two ENCODINGS. This is the knob the lp_gap
#                           column measures -- the indicator encoding
#                           contributes nothing to the LP relaxation, so its
#                           root bound is 0 and lp_gap is 1.0 by construction.
#   sparse | bigm_sparse    the same two programs, assembled through the matrix
#                           API. Identical rows and identical lp_gap; what
#                           changes is only how long the BUILD takes (~2x for
#                           indicator, ~5-7x for bigm). Worth using on the large
#                           end of a scaling ladder, where the per-cell
#                           addConstr loop starts to show up next to the solve.
FORM=${FORM:-}
BIGM=${BIGM:-}
case "${FORM:-indicator}" in
    indicator|bigm|sparse|bigm_sparse) ;;
    *) echo "ERROR: unknown FORM '$FORM'; pick from" >&2
       echo "       indicator,bigm,sparse,bigm_sparse" >&2; exit 1 ;;
esac
# A run folder is results/<stamp>_<study>/ and a shard writes results_shard<k>.csv
# into it, so two submissions differing only in the encoding would overwrite each
# other whenever they share a stamp. Give each encoding its own subtree by
# default (override with OUT=). To compare the two, submit twice and point the
# analysis at both folders -- every row carries a 'formulation' column.
if [ -n "$FORM" ]; then
    OUT=${OUT:-$PROJECT/results/$FORM}
else
    OUT=${OUT:-$PROJECT/results}
fi
MAXPAR=${MAXPAR:-10}                  # concurrent array tasks
WALL=${WALL:-24:00:00}                # per-task Slurm limit
CPUS=${CPUS:-4}                       # MUST be constant across a whole study
MEM=${MEM:-4g}                        # per cpu
# One timestamp for the WHOLE run. Every array task starts at a different minute,
# so if each computed its own the shards would land in different folders and could
# never be merged. Format YYYYMMDDHHMM -> results/<stamp>_<study>/
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d%H%M)}

cd "$PROJECT"                         # so SLURM_SUBMIT_DIR == project root
mkdir -p logs "$OUT"                  # Slurm will NOT create the log directory

# Fail here, not after printing a submission plan we cannot carry out.
if [ ! -d "$PROJECT/.venv-euler" ]; then
    echo "ERROR: no .venv-euler in $PROJECT -- run 'bash euler/setup_euler.sh' first" >&2
    exit 1
fi
MISSING=""
for f in euler/run_studies_array.sbatch euler/merge_studies.sbatch \
         "$PROJECT/run_studies.py" "$PROJECT/test.py"; do
    [ -f "$f" ] || MISSING="$MISSING $f"
done
if [ -n "$MISSING" ]; then
    echo "ERROR: missing file(s):$MISSING" >&2
    echo "       run_studies.py imports test.py BY PATH for the Scenario, the run-" >&2
    echo "       folder layout and the CSV schema, so BOTH must be in the project" >&2
    echo "       root. The .sbatch files travel with the .sh files -- commit and" >&2
    echo "       push the whole euler/ folder, then 'git pull' here." >&2
    exit 1
fi
# A stale run_studies.py is the most wasteful failure here: every array task dies
# in seconds with argparse exit code 2 and you only find out from the logs.
for flag in --studies --shard --merge --run-stamp --threads --gurobi-params \
            --lp-time-limit --band --formulation; do
    if ! grep -q -- "\"$flag\"" "$PROJECT/run_studies.py"; then
        echo "ERROR: $PROJECT/run_studies.py does not support $flag -- it is an" >&2
        echo "       older version than these job scripts expect." >&2
        echo "       Update it, then check:" >&2
        echo "         python run_studies.py --help | grep -E -- '--studies|--shard'" >&2
        exit 1
    fi
done
# An unrecognised flag in EXTRA is the cheapest way to lose a whole array:
# argparse exits 2 in seconds, every shard dies, and the only trace is a usage
# dump in logs/. Check EXTRA's long options against --help BEFORE submitting.
if [ -n "${EXTRA:-}" ]; then
    source "$PROJECT/.venv-euler/bin/activate" 2>/dev/null || true
    HELP=$(python run_studies.py --help 2>&1 || true)
    BADFLAGS=""
    for tok in ${EXTRA}; do
        case "$tok" in
          --*) printf '%s' "$HELP" | grep -q -- "$tok" || BADFLAGS="$BADFLAGS $tok" ;;
        esac
    done
    if [ -n "$BADFLAGS" ]; then
        echo "ERROR: EXTRA contains flag(s) run_studies.py does not accept:$BADFLAGS" >&2
        echo "       Every shard would die with argparse exit code 2. Check:" >&2
        echo "         python run_studies.py --help | grep -- '--sparse'" >&2
        echo "       (The sparse strengthening is spelled --sparse-cuts" >&2
        echo "        off|core|full, or equivalently --formulation" >&2
        echo "        indicator_cuts.)" >&2
        exit 1
    fi
fi
for s in ${STUDIES//,/ }; do
    case "$s" in
        scaling|horizon|heatmap|convergence) ;;
        *) echo "ERROR: unknown study '$s'; pick from scaling,horizon,heatmap,convergence" >&2
           exit 1 ;;
    esac
done

# CRLF in a batch script makes Slurm fail in confusing ways ("not found" for a
# file that exists, because the interpreter becomes /usr/bin/bash\r).
for f in euler/run_studies_array.sbatch euler/merge_studies.sbatch; do
    if grep -qU $'\r' "$f" 2>/dev/null; then
        echo "ERROR: $f has Windows (CRLF) line endings." >&2
        echo "       Fix with: sed -i 's/\r$//' $f   (and add a .gitattributes" >&2
        echo "       rule '*.sbatch text eol=lf' so it does not come back)" >&2
        exit 1
    fi
done

echo "project   : $PROJECT"
echo "studies   : $STUDIES"
echo "encoding  : ${FORM:-indicator (harness default)}${BIGM:+  bigM=$BIGM}"
for s in ${STUDIES//,/ }; do
    echo "run folder: $OUT/${RUN_STAMP}_${s}"
done
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
# Array tasks read the code when each task STARTS, not when you submit. Editing or
# pulling while the array is queued means later tasks run different code than
# earlier ones; the merge step detects this and warns.
echo "note      : do not edit the code or 'git pull' until the array has finished"

# The budget and the feasibility screen, from the harness itself. This is the one
# check worth reading before every submission: it catches a widened ladder that
# quadrupled the cost, and any (configuration, bound) pair that will come back
# 'infeasible' in milliseconds instead of producing a solve-time measurement.
echo
echo "=== plan (run_studies.py --plan) ================================="
set +e
source "$PROJECT/.venv-euler/bin/activate" 2>/dev/null
python run_studies.py --plan --studies "$STUDIES" \
    --time-limit "${SOLVE_TL:-300}" --mip-gap "${MIP_GAP:-1e-4}" \
    --plan-shards "$NSHARDS" ${FORM:+--formulation "$FORM"} ${EXTRA:-} 2>&1 \
    | grep -E "^base case|^combos|^seeds|^time limit|^  -> |^TOTAL|^Worst case|^With --shard|^WARNING|screen +note|no:"
PLAN_RC=${PIPESTATUS[0]}
set -e
if [ "$PLAN_RC" -ne 0 ]; then
    echo "WARNING --plan exited $PLAN_RC; submitting anyway, but check the CLI args" >&2
fi
echo "=================================================================="
echo

if [ "${CONFIRM:-0}" = "1" ]; then
    read -r -p "submit? [y/N] " ans
    case "$ans" in y|Y) ;; *) echo "aborted"; exit 0 ;; esac
fi

# NSHARDS is exported so the job never has to infer the array size itself.
ARRAY_ID=$(PROJECT=$PROJECT STUDIES=$STUDIES NAME=$NAME OUT=$OUT NSHARDS=$NSHARDS \
    RUN_STAMP=$RUN_STAMP SOLVE_TL="${SOLVE_TL:-300}" MIP_GAP="${MIP_GAP:-1e-4}" \
    LP_TL="${LP_TL:-60}" LP_RELAX="${LP_RELAX:-1}" \
    FORM="$FORM" BIGM="$BIGM" \
    GUROBI_PARAMS="${GUROBI_PARAMS:-}" EXTRA="${EXTRA:-}" \
    sbatch --parsable --array=0-$((NSHARDS - 1))%"$MAXPAR" \
           --time="$WALL" --cpus-per-task="$CPUS" --mem-per-cpu="$MEM" \
           euler/run_studies_array.sbatch)
echo "array job : $ARRAY_ID  ($NSHARDS shards, max $MAXPAR at once, $CPUS cpus, $WALL)"

MERGE_ID=$(PROJECT=$PROJECT STUDIES=$STUDIES NAME=$NAME OUT=$OUT \
    RUN_STAMP=$RUN_STAMP BAND="${BAND:-iqr}" \
    sbatch --parsable --dependency=afterany:"$ARRAY_ID" euler/merge_studies.sbatch)
echo "merge job : $MERGE_ID  (runs after the array finishes)"
echo
echo "watch with : squeue -u \$USER   /   myjobs -j $ARRAY_ID"
echo "results in : $OUT/${RUN_STAMP}_<study>/  (merged_summary.txt when done)"
echo
echo "re-merge without re-solving (e.g. to change the band):"
echo "  RUN_STAMP=$RUN_STAMP STUDIES=$STUDIES OUT=$OUT BAND=minmax sbatch euler/merge_studies.sbatch"
