#!/bin/bash
#SBATCH --job-name=fm-regr
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/regression_%j.out
#SBATCH --error=logs/regression_%j.err

# WP8: runs the regression check scripts of examples/regression/ that the merge kept, one by one.
set -uo pipefail

VENV=/cluster/home/lfiaschi/fleet_management/venv
echo "START $(date) job=$SLURM_JOB_ID node=$SLURMD_NODENAME pwd=$PWD"

module load stack/2024-06 gcc/12.2.0
module load python/3.12.8
module load stack/2024-06 gurobi/13.0.0
source "$VENV/bin/activate"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg

mkdir -p regression_output
cd regression_output

failed=0
for script in \
    check_gamma_convolution_quadrature.py \
    check_gamma_tail_bound.py \
    check_horizon_sweep_visualization.py \
    check_gamma_replay_validator.py \
    check_matched_ard_inputs.py \
    check_unrestricted_maintenance.py
do
    echo "=== $script"
    start=$(date +%s)
    if timeout 900 python "../examples/regression/$script"; then
        echo "=== $script: PASS ($(( $(date +%s) - start )) s)"
    else
        rc=$?
        echo "=== $script: FAIL exit=$rc ($(( $(date +%s) - start )) s)"
        failed=$((failed + 1))
    fi
done

echo "END $(date) failed=$failed"
exit "$failed"
