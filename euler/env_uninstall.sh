#!/bin/bash
#SBATCH --job-name=fm-uninstall
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --time=00:10:00
#SBATCH --output=logs/uninstall_%j.out
#SBATCH --error=logs/uninstall_%j.err

set -euo pipefail

VENV=/cluster/home/lfiaschi/fleet_management/venv
GUROBIPY_VERSION=13.0.0
PYTHON_VERSION=3.12.8
TRASH=/cluster/scratch/lfiaschi/trash

echo "START $(date) job=$SLURM_JOB_ID node=$SLURMD_NODENAME pwd=$PWD"

module load stack/2024-06 gcc/12.2.0
module load python/3.12.8
module load stack/2024-06 gurobi/13.0.0

if [ ! -d "$VENV" ]; then
    echo "No environment at $VENV: nothing to do"
    echo "END $(date) exit=0"
    exit 0
fi

# Decide whether the environment is outdated (failing commands must not stop the script)
reason=""
py_version=""
if ! py_version=$("$VENV/bin/python" -c 'import platform; print(platform.python_version())' 2>/dev/null); then
    reason="the Python of the environment cannot run"
elif [ "$py_version" != "$PYTHON_VERSION" ]; then
    reason="Python is $py_version, expected $PYTHON_VERSION"
else
    gp_version=$("$VENV/bin/python" -m pip show gurobipy 2>/dev/null | sed -n 's/^Version: //p' || true)
    if [ -z "$gp_version" ]; then
        reason="gurobipy is missing"
    elif [ "$gp_version" != "$GUROBIPY_VERSION" ]; then
        reason="gurobipy is $gp_version, expected $GUROBIPY_VERSION"
    fi
fi

if [ -n "$reason" ]; then
    echo "Environment is outdated: $reason"
    mkdir -p "$TRASH"
    dest="$TRASH/venv_$(date +%Y%m%d_%H%M%S)"
    mv "$VENV" "$dest"
    echo "Moved $VENV to $dest"
else
    echo "Environment is up to date: removing the package only"
    "$VENV/bin/python" -m pip uninstall -y fleet-management
fi

echo "END $(date) exit=0"
