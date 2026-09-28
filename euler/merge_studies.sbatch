#!/usr/bin/bash
#SBATCH --job-name=fleet_studies_merge
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8g
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Reduce step. A shard holds a slice of the design points, so it can neither
# aggregate over seeds (the median/IQR bands) nor compare consecutive H values
# (the first-period flip metric) -- both of those live here, together with the
# figures. Solves nothing, so it is cheap. Submitted with a dependency on the
# array.
#
# 8 GB rather than the array's 4: the merge holds every row of every shard in
# memory at once and matplotlib is not frugal. Still trivial next to a solve.
set -euo pipefail

PROJECT=${PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}
STUDIES=${STUDIES:-scaling}
RUN_STAMP=${RUN_STAMP:?RUN_STAMP must be exported by euler/submit_studies.sh}
NAME=${NAME:-studies}
OUT=${OUT:-$PROJECT/results}
# Cosmetic knobs for the figures; they change no data, so the merge can be re-run
# with different values as often as you like without touching the cluster again.
BAND=${BAND:-iqr}                     # iqr | minmax | std
EXTRA=${EXTRA:-}

if ! command -v module >/dev/null 2>&1; then
    for init in /etc/profile.d/modules.sh \
                /cluster/apps/modules/init/bash \
                /usr/share/lmod/lmod/init/bash; do
        [ -f "$init" ] && . "$init" && break
    done
fi
set +e
module purge
# No gurobi module: the merge never builds a model, so it must not hold a licence
# token while it draws pictures.
module load stack/2024-06 python/3.12.8
set -e
source "$PROJECT/.venv-euler/bin/activate"
cd "$PROJECT"

echo "merging studies=$STUDIES  stamp=$RUN_STAMP  band=$BAND"
python run_studies.py --merge --studies "$STUDIES" --name "$NAME" --out "$OUT" \
                      --run-stamp "$RUN_STAMP" --band "$BAND" $EXTRA

echo
for S in ${STUDIES//,/ }; do
    D="$OUT/${RUN_STAMP}_${S}"
    echo "== $S -> $D"
    ls -1 "$D"/merged_* 2>/dev/null || echo "   (nothing merged; check the array logs)"
done
