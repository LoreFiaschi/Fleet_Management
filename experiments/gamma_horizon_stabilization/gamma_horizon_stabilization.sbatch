#!/usr/bin/bash
#SBATCH --job-name=gamma-horizon-final
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --output=runs/slurm-%j.out
#SBATCH --error=runs/slurm-%j.err

set -euo pipefail

SUBMIT_DIR=${SLURM_SUBMIT_DIR:-$PWD}
PROJECT_ROOT=$(git -C "$SUBMIT_DIR" rev-parse --show-toplevel)
EXPERIMENT_DIR="$PROJECT_ROOT/experiments/gamma_horizon_stabilization"
RUN_ID="horizon_${SLURM_JOB_ID}"
RUN_DIR="$EXPERIMENT_DIR/runs/$RUN_ID"
INPUT="$EXPERIMENT_DIR/input/gamma_horizon_stabilization.yaml"
REPORT="$RUN_DIR/horizon_sweep.yaml"
ARCHIVE="$EXPERIMENT_DIR/runs/gamma_horizon_stabilization_${SLURM_JOB_ID}.tar.gz"

mkdir -p "$RUN_DIR/source_snapshot"

finalize() {
    status=$?
    trap - EXIT
    set +e
    date --iso-8601=seconds > "$RUN_DIR/finished_at.txt"
    printf '%s\n' "$status" > "$RUN_DIR/exit_status.txt"
    (
        cd "$RUN_DIR"
        find . -type f ! -name SHA256SUMS -print0 \
            | sort -z \
            | xargs -0 sha256sum > SHA256SUMS
    )
    tar -czf "$ARCHIVE" -C "$EXPERIMENT_DIR/runs" "$RUN_ID"
    sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"
    echo "Run directory : $RUN_DIR"
    echo "Archive       : $ARCHIVE"
    echo "Exit status   : $status"
    exit "$status"
}
trap finalize EXIT

cd "$PROJECT_ROOT"

module purge
module load stack/2024-06 python/3.12.8
module load gurobi/13.0.0
source .venv/bin/activate

GUROBI_VERSION=$(python -c "import gurobipy as gp; print('.'.join(map(str, gp.gurobi.version())))")

if [[ -n "$(git status --porcelain)" ]]; then
    echo "ERROR: repository is dirty; commit or remove local changes first." >&2
    git status --short >&2
    exit 2
fi

cp "$INPUT" "$RUN_DIR/input.yaml"
cp "$EXPERIMENT_DIR/gamma_horizon_stabilization.sbatch" "$RUN_DIR/job.sbatch"
cp "$EXPERIMENT_DIR/plot_result.py" "$RUN_DIR/plot_result.py"
cp "$EXPERIMENT_DIR/summarize_result.py" "$RUN_DIR/summarize_result.py"
cp examples/regression/run_horizon_sweep.py "$RUN_DIR/run_horizon_sweep.py"
cp pyproject.toml "$RUN_DIR/source_snapshot/pyproject.toml"
cp src/fleet_management/horizon_sweep.py "$RUN_DIR/source_snapshot/horizon_sweep.py"
cp src/fleet_management/degradation_model/base.py "$RUN_DIR/source_snapshot/base.py"
cp src/fleet_management/degradation_model/gamma_utils/gamma_diagnostics.py \
   "$RUN_DIR/source_snapshot/gamma_diagnostics.py"
cp src/fleet_management/utils/mixed_plotter.py \
   "$RUN_DIR/source_snapshot/mixed_plotter.py"

{
    echo "experiment=gamma_horizon_stabilization"
    echo "job_id=$SLURM_JOB_ID"
    echo "started_at=$(date --iso-8601=seconds)"
    echo "host=$(hostname)"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "git_branch=$(git branch --show-current)"
    echo "git_dirty=no"
    echo "python=$(python --version 2>&1)"
    echo "gurobi=$GUROBI_VERSION"
    echo "fixed=F4_M1_L1_H1_4"
    echo "varied=H2_4_to_64_step_4"
    echo "per_case_time_limit_seconds=3300"
    echo "solver_mip_gap=0"
    echo "presentation_gap_qualification=0.05"
} > "$RUN_DIR/metadata.txt"

module list 2> "$RUN_DIR/modules.txt"
python -m pip freeze > "$RUN_DIR/python_packages.txt"

python - "$INPUT" <<'PY' > "$RUN_DIR/preflight.txt"
from pathlib import Path
import sys
import yaml
from fleet_management.config import load_config

path = Path(sys.argv[1])
cfg = load_config(yaml.safe_load(path.read_text(encoding="utf-8")))
assert (cfg.F, cfg.M, cfg.L, cfg.H1, cfg.H2) == (4, 1, 1, 4, 4)
assert str(cfg.model[0, 0]) == "gamma"
assert str(cfg.repair_model[0, 0]) == "ardinf"
print(f"PASS {path}")
print(f"F={cfg.F}, M={cfg.M}, L={cfg.L}, H1={cfg.H1}, base_H2={cfg.H2}")
PY

echo "============================================================"
echo "FINAL GAMMA OPERATING-HORIZON SWEEP"
echo "============================================================"
cat "$RUN_DIR/metadata.txt"
echo "run_directory=$RUN_DIR"

/usr/bin/time -v python "$RUN_DIR/run_horizon_sweep.py" \
    "$RUN_DIR/input.yaml" \
    "$REPORT" \
    --h2-range 4 64 \
    --h2-step 4 \
    --maximum-stopping-gap 0.05 \
    > "$RUN_DIR/runner.out" \
    2> "$RUN_DIR/runner.err"

python "$RUN_DIR/summarize_result.py" "$REPORT" "$RUN_DIR" \
    > "$RUN_DIR/postprocess.out" 2> "$RUN_DIR/postprocess.err"

MPLBACKEND=Agg python "$RUN_DIR/plot_result.py" \
    "$REPORT" "$RUN_DIR/gamma_horizon_stabilization.png" \
    >> "$RUN_DIR/postprocess.out" 2>> "$RUN_DIR/postprocess.err"

cat "$RUN_DIR/postprocess.out"
