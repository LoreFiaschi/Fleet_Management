#!/usr/bin/bash
#SBATCH --job-name=vbz-greedy-start
#SBATCH --time=02:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=1G
#SBATCH --output=runs/vbz-greedy-start-%j.out
#SBATCH --error=runs/vbz-greedy-start-%j.err

set -Eeuo pipefail

REPO_ROOT=${SLURM_SUBMIT_DIR:-$PWD}
EXPERIMENT_REL=experiments/vbz_greedy_warm_start
EXPERIMENT_DIR="$REPO_ROOT/$EXPERIMENT_REL"
RUNS_DIR="$EXPERIMENT_DIR/runs"
JOB_ID=${SLURM_JOB_ID:-manual}
RUN_NAME="vbz_greedy_warm_start_${JOB_ID}"
RUN_DIR="$RUNS_DIR/$RUN_NAME"
ARCHIVE="$RUNS_DIR/${RUN_NAME}.tar.gz"

mkdir -p "$RUN_DIR" "$REPO_ROOT/runs"

finish() {
    status=$?
    trap - EXIT
    set +e
    date --iso-8601=seconds > "$RUN_DIR/finished_at.txt"
    printf '%s\n' "$status" > "$RUN_DIR/exit_status.txt"
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

cp "$EXPERIMENT_DIR/input.yaml" "$RUN_DIR/input_source.yaml"
cp "$EXPERIMENT_DIR/run_comparison.py" "$RUN_DIR/run_comparison.py"
cp "$EXPERIMENT_DIR/README.md" "$RUN_DIR/README.md"
cp "$EXPERIMENT_DIR/vbz_greedy_warm_start.sbatch" "$RUN_DIR/job.sbatch"

{
    echo "job_id=$JOB_ID"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
    echo "started_at=$(date --iso-8601=seconds)"
    echo "slurm_cpus=${SLURM_CPUS_PER_TASK:-unset}"
    echo "comparison=baseline_then_greedy"
} > "$RUN_DIR/metadata.txt"

PYTHONPATH="$REPO_ROOT/src" python - <<'PY' > "$RUN_DIR/preflight.txt"
import os
import yaml
from fleet_management.config import load_config
from fleet_management.greedy_warm_start import build_greedy_warm_start

path = "experiments/vbz_greedy_warm_start/input.yaml"
raw = yaml.safe_load(open(path, encoding="utf-8"))
cfg = load_config(raw)
start, report = build_greedy_warm_start(cfg)
allocated = int(os.environ["SLURM_CPUS_PER_TASK"])
assert (cfg.F, cfg.M, cfg.L, cfg.H1, cfg.H2) == (4, 2, 2, 4, 12)
assert allocated == int(raw["gurobi_params"]["Threads"]) == 32
assert report.feasible_assignment and report.artificial_assignments == 0
assert report.repeatability_feasible and report.repeatability_failures == 0
assert report.repairs > 0
print("PASS dimensions and allocation")
print(report.as_dict())
PY

PYTHONPATH="$REPO_ROOT/src" python "$EXPERIMENT_DIR/run_comparison.py" \
    "$EXPERIMENT_DIR/input.yaml" "$RUN_DIR" \
    > "$RUN_DIR/runner.out" 2> "$RUN_DIR/runner.err"
