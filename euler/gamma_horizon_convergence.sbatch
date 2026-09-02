#!/usr/bin/bash
#SBATCH --job-name=gamma-horizon-convergence
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --output=gamma-horizon-convergence-%j.out
#SBATCH --error=gamma-horizon-convergence-%j.err

set -euo pipefail

PROJECT=${SLURM_SUBMIT_DIR:-$PWD}
INPUT=${INPUT:-input/gamma_horizon_euler_convergence.yaml}

# These defaults perform a coarse convergence search. They can be overridden
# at submission without editing this file; see the commands printed below.
H2_START=${H2_START:-4}
H2_STOP=${H2_STOP:-40}
H2_STEP=${H2_STEP:-4}
GRADIENT_TOLERANCE=${GRADIENT_TOLERANCE:-0.001}
FLAT_GRADIENTS=${FLAT_GRADIENTS:-2}
MINIMUM_CASES=${MINIMUM_CASES:-4}
MAXIMUM_STOPPING_GAP=${MAXIMUM_STOPPING_GAP:-0.05}

cd "$PROJECT"

module purge
module load stack/2024-06 python/3.12.8
module load gurobi/13.0.0

source .venv/bin/activate
mkdir -p results

REPORT="results/gamma_horizon_convergence_${SLURM_JOB_ID}.yaml"
INPUT_SNAPSHOT="results/gamma_horizon_convergence_${SLURM_JOB_ID}_input.yaml"
cp "$INPUT" "$INPUT_SNAPSHOT"

echo "============================================================"
echo "CONTROLLED GAMMA HORIZON-CONVERGENCE SWEEP"
echo "============================================================"
echo "Date          : $(date --iso-8601=seconds)"
echo "Host          : $(hostname)"
echo "Git commit    : $(git rev-parse HEAD)"
echo "Git dirty     : $(test -n "$(git status --porcelain)" && echo yes || echo no)"
echo "Input         : $INPUT"
echo "Input snapshot: $INPUT_SNAPSHOT"
echo "Fixed         : F=4, M=1, L=1, H1=4"
echo "Varied        : H2=$H2_START:$H2_STEP:$H2_STOP"
echo "Stop rule     : |relative gradient| <= $GRADIENT_TOLERANCE"
echo "                for $FLAT_GRADIENTS comparisons, or cost increase"
echo "Maximum gap   : $MAXIMUM_STOPPING_GAP for gradient-based stopping"
echo "Per-case limit: 900 s (from the YAML)"
echo "Report        : $REPORT"

/usr/bin/time -v python examples/regression/run_horizon_sweep.py \
    "$INPUT" \
    "$REPORT" \
    --h2-range "$H2_START" "$H2_STOP" \
    --h2-step "$H2_STEP" \
    --stop-on-gradient \
    --gradient-tolerance "$GRADIENT_TOLERANCE" \
    --flat-gradients "$FLAT_GRADIENTS" \
    --minimum-cases "$MINIMUM_CASES" \
    --maximum-stopping-gap "$MAXIMUM_STOPPING_GAP"

python - "$REPORT" <<'PY'
from pathlib import Path
import sys
import yaml

path = Path(sys.argv[1])
report = yaml.safe_load(path.read_text(encoding="utf-8"))

print("\nRESULT SUMMARY")
print(
    "H2   T   status          J_op/H2    best bound   MIP gap   "
    "gradient/H2   continuous  integer  linear rows  optimizer_s"
)
for row in report["cases"]:
    formulation = row["formulation"]

    def number(value, spec=".6g"):
        return "-" if value is None else format(float(value), spec)

    print(
        f"{row['H2']:>2}  {row['T']:>2}  {str(row['status']):<14} "
        f"{number(row.get('J_op_average')):>10}  "
        f"{number(row.get('objective_bound')):>10}  "
        f"{number(row.get('mip_gap'), '.2%'):>8}  "
        f"{number(row.get('relative_cost_gradient_per_H2'), '.3%'):>11}  "
        f"{formulation['continuous_variables']:>10}  "
        f"{formulation['integer_variables']:>7}  "
        f"{formulation['linear_constraints']:>11}  "
        f"{number(row.get('optimizer_seconds'), '.3f'):>11}"
    )

print("evaluated H2       :", report["evaluated_operating_horizons"])
print("stopping reason    :", report["stopping_rule"]["reason"])
print("best proven H2     :", report["best_proven_H2"])
print("best feasible H2   :", report["best_feasible_H2"])
print("best feasible gap  :", next(
    (
        row.get("mip_gap") for row in report["cases"]
        if row["H2"] == report["best_feasible_H2"]
    ),
    None,
))
print("formulation growth :", report["formulation_growth"])
print("report             :", path)
PY

echo "Finished      : $(date --iso-8601=seconds)"
