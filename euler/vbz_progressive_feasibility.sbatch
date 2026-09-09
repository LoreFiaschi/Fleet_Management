#!/usr/bin/bash
#SBATCH --job-name=vbz-feasibility-ladder
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=vbz-feasibility-ladder-%j.out
#SBATCH --error=vbz-feasibility-ladder-%j.err

set -euo pipefail

PROJECT=${SLURM_SUBMIT_DIR:-$PWD}
cd "$PROJECT"

module purge
module load stack/2024-06 python/3.12.8
module load gurobi/13.0.0
source .venv/bin/activate

ROOT="results/vbz_progressive_${SLURM_JOB_ID}"
SUMMARY="results/vbz_progressive_${SLURM_JOB_ID}.yaml"
SNAPSHOTS="${ROOT}/inputs"
mkdir -p "$ROOT" "$SNAPSHOTS"

INPUTS=(
  examples/regression/vbz_progressive/vbz_00_reference.yaml
  examples/regression/vbz_progressive/vbz_01_full_horizon.yaml
  examples/regression/vbz_progressive/vbz_02_four_components.yaml
  examples/regression/vbz_progressive/vbz_03_regional_fleet.yaml
  examples/regression/vbz_progressive/vbz_04_three_missions.yaml
  examples/regression/vbz_progressive/vbz_05_oberes_glattal.yaml
)

cp "${INPUTS[@]}" "$SNAPSHOTS/"

echo "============================================================"
echo "PROGRESSIVE VBZ REGIONAL FEASIBILITY LADDER"
echo "============================================================"
echo "Date       : $(date --iso-8601=seconds)"
echo "Host       : $(hostname)"
echo "Git commit : $(git rev-parse HEAD)"
echo "Git dirty  : $(test -n "$(git status --porcelain)" && echo yes || echo no)"
echo "Summary    : $SUMMARY"
echo "Results    : $ROOT"

/usr/bin/time -v python examples/regression/run_progressive_vbz_feasibility.py \
  "${INPUTS[@]}" \
  --output-dir "$ROOT" \
  --summary "$SUMMARY"

echo "Finished   : $(date --iso-8601=seconds)"
