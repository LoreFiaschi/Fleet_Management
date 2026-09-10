#!/usr/bin/bash
#SBATCH --job-name=vbz-F8-transfer
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --output=runs/slurm-%j.out
#SBATCH --error=runs/slurm-%j.err

set -euo pipefail

SUBMIT_DIR=${SLURM_SUBMIT_DIR:-$PWD}
PROJECT=$(git -C "$SUBMIT_DIR" rev-parse --show-toplevel)
SCRIPT_DIR="$PROJECT/experiments/vbz_F8_feasibility_transfer"
JOB_ID=${SLURM_JOB_ID:-manual}
RUN_DIR="$SCRIPT_DIR/runs/f8_transfer_${JOB_ID}"

mkdir -p "$RUN_DIR"
cd "$PROJECT"

module purge
module load stack/2024-06 python/3.12.8
module load gurobi/13.0.0
source "$PROJECT/.venv/bin/activate"

cp "$SCRIPT_DIR/inputs/vbz_F6_M1_L4_H2_48.yaml" "$RUN_DIR/"
cp "$SCRIPT_DIR/inputs/vbz_F8_M1_L4_H2_48.yaml" "$RUN_DIR/"
cp "$SCRIPT_DIR/starts/vbz_F6_M1_L4_H2_48_result.yaml" "$RUN_DIR/"
cp "$SCRIPT_DIR/run_f8_feasibility_transfer.py" "$RUN_DIR/"
cp "$SCRIPT_DIR/vbz_F8_feasibility_transfer.sbatch" "$RUN_DIR/"

git rev-parse HEAD > "$RUN_DIR/git_commit.txt"
git status --short > "$RUN_DIR/git_status.txt"
git diff > "$RUN_DIR/uncommitted.patch"
python --version > "$RUN_DIR/python_version.txt" 2>&1

python "$SCRIPT_DIR/run_f8_feasibility_transfer.py" \
  --f6-input "$SCRIPT_DIR/inputs/vbz_F6_M1_L4_H2_48.yaml" \
  --f8-input "$SCRIPT_DIR/inputs/vbz_F8_M1_L4_H2_48.yaml" \
  --f6-result "$SCRIPT_DIR/starts/vbz_F6_M1_L4_H2_48_result.yaml" \
  --output-dir "$RUN_DIR" \
  --summary "$RUN_DIR/f8_feasibility_transfer_summary.yaml"

tar -czf "$SCRIPT_DIR/runs/f8_transfer_${JOB_ID}.tar.gz" -C "$RUN_DIR" .

echo "Run directory: $RUN_DIR"
echo "Archive      : $SCRIPT_DIR/runs/f8_transfer_${JOB_ID}.tar.gz"
