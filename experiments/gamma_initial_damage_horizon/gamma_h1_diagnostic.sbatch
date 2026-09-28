#!/usr/bin/bash
#SBATCH --job-name=gamma-H1-damage
#SBATCH --time=03:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=1G
#SBATCH --output=runs/gamma-H1-damage-%j.out
#SBATCH --error=runs/gamma-H1-damage-%j.err

set -Eeuo pipefail

REPO_ROOT=${SLURM_SUBMIT_DIR:-$PWD}
EXPERIMENT_REL=experiments/gamma_initial_damage_horizon
EXPERIMENT_DIR="$REPO_ROOT/$EXPERIMENT_REL"
RUNS_DIR="$EXPERIMENT_DIR/runs"
JOB_ID=${SLURM_JOB_ID:-manual}
RUN_NAME="gamma_h1_initial_damage_${JOB_ID}"
RUN_DIR="$RUNS_DIR/$RUN_NAME"
ARCHIVE="$RUNS_DIR/${RUN_NAME}.tar.gz"
INPUT_NAME=horizon_initial_damage.yaml
REPORT="$RUN_DIR/h1_diagnostic_report.yaml"

mkdir -p "$RUN_DIR/source_snapshot" "$REPO_ROOT/runs"

finish() {
    status=$?
    trap - EXIT
    set +e
    date --iso-8601=seconds > "$RUN_DIR/finished_at.txt"
    printf '%s\n' "$status" > "$RUN_DIR/exit_status.txt"
    (cd "$RUN_DIR" && find . -type f ! -name SHA256SUMS -print0 \
        | sort -z | xargs -0 -r sha256sum > SHA256SUMS)
    tar -czf "$ARCHIVE" -C "$RUNS_DIR" "$RUN_NAME"
    sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"
    echo "Run directory: $RUN_DIR"
    echo "Archive: $ARCHIVE"
    echo "Exit status: $status"
    exit "$status"
}
trap finish EXIT

cd "$REPO_ROOT"
module purge
module load stack/2024-06 python/3.12.8
module load gurobi/13.0.0
source "$REPO_ROOT/.venv/bin/activate"
export PYTHONUNBUFFERED=1

if [[ ! -f pyproject.toml || ! -d "$EXPERIMENT_DIR/input" ]]; then
    echo "Submit from the Fleet_Management repository root." >&2
    exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "ERROR: tracked files are dirty; commit or restore them first." >&2
    git status --short >&2
    exit 2
fi

PYTHON="$REPO_ROOT/.venv/bin/python"
INPUT="$EXPERIMENT_DIR/input/$INPUT_NAME"

cp "$EXPERIMENT_DIR/gamma_h1_diagnostic.sbatch" "$RUN_DIR/job.sbatch"
cp "$EXPERIMENT_DIR/README.md" "$RUN_DIR/README.md"
cp "$EXPERIMENT_DIR/preflight.py" "$RUN_DIR/preflight.py"
cp "$EXPERIMENT_DIR/summarize.py" "$RUN_DIR/summarize.py"
cp "$EXPERIMENT_DIR/plot_results.py" "$RUN_DIR/plot_results.py"
cp "$INPUT" "$RUN_DIR/$INPUT_NAME"
cp examples/regression/run_horizon_sweep.py "$RUN_DIR/run_horizon_sweep.py"
cp pyproject.toml "$RUN_DIR/source_snapshot/pyproject.toml"
cp src/fleet_management/config.py "$RUN_DIR/source_snapshot/config.py"
cp src/fleet_management/solver.py "$RUN_DIR/source_snapshot/solver.py"
cp src/fleet_management/horizon_sweep.py "$RUN_DIR/source_snapshot/horizon_sweep.py"
cp src/fleet_management/degradation_model/base.py "$RUN_DIR/source_snapshot/base.py"
cp src/fleet_management/degradation_model/gamma.py "$RUN_DIR/source_snapshot/gamma.py"
cp src/fleet_management/degradation_model/rainflow.py "$RUN_DIR/source_snapshot/rainflow.py"

{
    echo "experiment=gamma_h1_initial_damage"
    echo "job_id=$JOB_ID"
    echo "host=$(hostname)"
    echo "started_at=$(date --iso-8601=seconds)"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
    echo "H1_candidates=2,4,6,8,10,12"
    echo "H2_fixed=24"
    echo "per_case_time_limit_seconds=1500"
    echo "phase_I_time_limit_seconds=120"
    echo "threads=32"
} > "$RUN_DIR/metadata.txt"

module list > "$RUN_DIR/modules.txt" 2>&1 || true
"$PYTHON" -m pip freeze > "$RUN_DIR/python_packages.txt" 2>&1 || true
"$PYTHON" "$RUN_DIR/preflight.py" "$INPUT" > "$RUN_DIR/preflight.txt" 2>&1
cat "$RUN_DIR/preflight.txt"

/usr/bin/time -v "$PYTHON" "$RUN_DIR/run_horizon_sweep.py" \
    "$RUN_DIR/$INPUT_NAME" "$REPORT" \
    --h1 2 4 6 8 10 12 \
    --h2 24 \
    --warm-start \
    --evaluation-horizon 52 \
    > "$RUN_DIR/runner.out" 2> "$RUN_DIR/runner.err"

"$PYTHON" "$RUN_DIR/summarize.py" "$REPORT" "$RUN_DIR" \
    > "$RUN_DIR/postprocess.out" 2> "$RUN_DIR/postprocess.err"
MPLBACKEND=Agg "$PYTHON" "$RUN_DIR/plot_results.py" \
    "$REPORT" "$RUN_DIR/h1_diagnostic.png" --kind h1 \
    >> "$RUN_DIR/postprocess.out" 2>> "$RUN_DIR/postprocess.err"
cat "$RUN_DIR/postprocess.out"
