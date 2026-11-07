#!/bin/bash
#SBATCH --job-name=fm-tests
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=02:00:00
#SBATCH --output=logs/tests_%j.out
#SBATCH --error=logs/tests_%j.err

set -euo pipefail

VENV=/cluster/home/lfiaschi/fleet_management/venv

echo "START $(date) job=$SLURM_JOB_ID node=$SLURMD_NODENAME pwd=$PWD"

module load stack/2024-06 gcc/12.2.0
module load python/3.12.8
module load stack/2024-06 gurobi/13.0.0

if [ ! -f "$VENV/bin/activate" ]; then
    echo "Environment $VENV is missing: run env_install.sh first"
    echo "END $(date) exit=2"
    exit 2
fi
source "$VENV/bin/activate"

# the code of the run folder is the one tested
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -c "import fleet_management, sys; print(fleet_management.__file__)"

if ! python -c "import gurobipy as gp; print(gp.gurobi.version()); m = gp.Model(); m.dispose()"; then
    echo "Gurobi licence check failed"
    echo "END $(date) exit=2"
    exit 2
fi

# arguments select tests (for example: sbatch run_tests.sh tests/test_spec_inputs.py); none means the whole suite
targets=("$@")
if [ ${#targets[@]} -eq 0 ]; then
    targets=(tests)
fi

# pytest may fail: keep its exit code instead of stopping the script
rc=0
python -m pytest "${targets[@]}" -v -rA --junitxml="logs/tests_${SLURM_JOB_ID}.xml" || rc=$?

echo "END $(date) exit=$rc"
exit "$rc"
