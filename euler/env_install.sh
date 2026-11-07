#!/bin/bash
#SBATCH --job-name=fm-install
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/install_%j.out
#SBATCH --error=logs/install_%j.err

set -euo pipefail

VENV=/cluster/home/lfiaschi/fleet_management/venv
GUROBIPY_VERSION=13.0.0

echo "START $(date) job=$SLURM_JOB_ID node=$SLURMD_NODENAME pwd=$PWD"

module load stack/2024-06 gcc/12.2.0
module load python/3.12.8
module load stack/2024-06 gurobi/13.0.0
# internet for pip on a compute node
module load eth_proxy

if [ ! -x "$VENV/bin/python" ]; then
    python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"

python -m pip install --upgrade pip
python -m pip install "gurobipy==$GUROBIPY_VERSION"
# installs the package of the folder where the job is submitted
python -m pip install ".[dev]"

python --version
python -m pip show gurobipy fleet-management | grep -E '^(Name|Version):'
python -m pip list

echo "END $(date) exit=0"
