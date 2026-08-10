#!/usr/bin/bash
#SBATCH --job-name=bound_merge
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4g
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Reduce step: combines every shard folder, evaluates H1/H2/H3 once, draws the
# plots. Solves nothing, so it is cheap. Submitted with a dependency on the array.
set -euo pipefail

PROJECT=${PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}
TEST=${TEST:-sweep}
NAME=${NAME:-bound_tightness}
OUT=${OUT:-$PROJECT/results}

module purge
module load stack/2024-06 python/3.12.8
source "$PROJECT/.venv-euler/bin/activate"
cd "$PROJECT"

python test.py --merge --tests "$TEST" --name "$NAME" --out "$OUT"
