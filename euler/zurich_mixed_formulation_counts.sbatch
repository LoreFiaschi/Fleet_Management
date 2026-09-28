#!/usr/bin/bash
#SBATCH --job-name=zurich-mixed-counts
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --output=zurich-mixed-counts-%j.out
#SBATCH --error=zurich-mixed-counts-%j.err

set -euo pipefail

PROJECT=${SLURM_SUBMIT_DIR:-$PWD}
REGIONAL_INPUT=${REGIONAL_INPUT:-input/zurich_oberes_glattal_mixed.yaml}
LARGE_INPUT=${LARGE_INPUT:-input/zurich_garage_hardau_mixed.yaml}

cd "$PROJECT"

module purge
module load stack/2024-06 python/3.12.8
module load gurobi/13.0.0

source .venv/bin/activate
mkdir -p results

REPORT="results/zurich_mixed_formulation_counts_${SLURM_JOB_ID}.yaml"
SNAPSHOT_DIR="results/zurich_mixed_formulation_counts_${SLURM_JOB_ID}_inputs"
mkdir -p "$SNAPSHOT_DIR"
cp "$REGIONAL_INPUT" "$SNAPSHOT_DIR/"
cp "$LARGE_INPUT" "$SNAPSHOT_DIR/"

echo "BUILD-ONLY ZURICH MIXED-FORMULATION REPORT"
echo "Date          : $(date --iso-8601=seconds)"
echo "Host          : $(hostname)"
echo "Git commit    : $(git rev-parse HEAD)"
echo "Git dirty     : $(test -n "$(git status --porcelain)" && echo yes || echo no)"
echo "Regional input: $REGIONAL_INPUT"
echo "Large input   : $LARGE_INPUT"
echo "Report        : $REPORT"
echo "Optimization  : disabled"

/usr/bin/time -v python examples/regression/report_mixed_formulation_size.py \
    "$REGIONAL_INPUT" \
    "$LARGE_INPUT" \
    --output "$REPORT"

echo "Finished      : $(date --iso-8601=seconds)"
