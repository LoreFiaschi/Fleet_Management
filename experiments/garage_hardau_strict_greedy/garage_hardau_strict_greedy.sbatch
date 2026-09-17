#!/usr/bin/bash
#SBATCH --job-name=hardau-strict-greedy
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=2G
#SBATCH --output=runs/hardau-strict-greedy-%j.out
#SBATCH --error=runs/hardau-strict-greedy-%j.err

set -Eeuo pipefail

REPO_ROOT=${SLURM_SUBMIT_DIR:-$PWD}
EXPERIMENT_REL=experiments/garage_hardau_strict_greedy
EXPERIMENT_DIR="$REPO_ROOT/$EXPERIMENT_REL"
RUNS_DIR="$EXPERIMENT_DIR/runs"
JOB_ID=${SLURM_JOB_ID:-manual}
RUN_NAME="garage_hardau_strict_greedy_${JOB_ID}"
RUN_DIR="$RUNS_DIR/$RUN_NAME"
ARCHIVE="$RUNS_DIR/${RUN_NAME}.tar.gz"

mkdir -p "$RUN_DIR/source_snapshot" "$REPO_ROOT/runs"

finish() {
    status=$?
    trap - EXIT
    set +e
    date --iso-8601=seconds > "$RUN_DIR/finished_at.txt"
    printf '%s\n' "$status" > "$RUN_DIR/exit_status.txt"
    (
        cd "$RUN_DIR" || exit 0
        find . -type f ! -name SHA256SUMS -print0 \
            | sort -z | xargs -0 -r sha256sum > SHA256SUMS
    )
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
cp "$EXPERIMENT_DIR/run_case.py" "$RUN_DIR/run_case.py"
cp "$EXPERIMENT_DIR/README.md" "$RUN_DIR/README.md"
cp "$EXPERIMENT_DIR/garage_hardau_strict_greedy.sbatch" "$RUN_DIR/job.sbatch"
cp pyproject.toml "$RUN_DIR/source_snapshot/pyproject.toml"
cp src/fleet_management/config.py "$RUN_DIR/source_snapshot/config.py"
cp src/fleet_management/solver.py "$RUN_DIR/source_snapshot/solver.py"
cp src/fleet_management/greedy_warm_start.py \
    "$RUN_DIR/source_snapshot/greedy_warm_start.py"
cp src/fleet_management/degradation_model/base.py \
    "$RUN_DIR/source_snapshot/base.py"
cp src/fleet_management/degradation_model/gamma.py \
    "$RUN_DIR/source_snapshot/gamma.py"
cp src/fleet_management/degradation_model/rainflow.py \
    "$RUN_DIR/source_snapshot/rainflow.py"

{
    echo "job_id=$JOB_ID"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
    echo "started_at=$(date --iso-8601=seconds)"
    echo "host=$(hostname)"
    echo "slurm_cpus=${SLURM_CPUS_PER_TASK:-unset}"
    echo "slurm_memory_per_cpu=${SLURM_MEM_PER_CPU:-unset}"
    echo "purpose=strict feasibility first, then incumbent/bound improvement"
} > "$RUN_DIR/metadata.txt"

module list > "$RUN_DIR/modules.txt" 2>&1 || true
"$REPO_ROOT/.venv/bin/python" -m pip freeze \
    > "$RUN_DIR/python_packages.txt" 2>&1 || true

PYTHONPATH="$REPO_ROOT/src" python - <<'PY' > "$RUN_DIR/preflight.txt"
import os
import yaml
from fleet_management.config import load_config

path = "experiments/garage_hardau_strict_greedy/input.yaml"
raw = yaml.safe_load(open(path, encoding="utf-8"))
cfg = load_config(raw)
assert (cfg.F, cfg.M, cfg.L, cfg.H1, cfg.H2, cfg.T) == (65, 13, 4, 4, 48, 52)
assert set(cfg.tau.ravel()) == {1.0}
assert set(cfg.epsilon.ravel()) == {0.0001}
assert raw["reliability_impl"] == "tangent"
assert int(raw["gurobi_params"]["Threads"]) == int(os.environ["SLURM_CPUS_PER_TASK"]) == 32
assert int(raw["time_limit"]) == 54000
print("PASS strict Hardau-inspired dimensions, safety and allocation")
print(f"F={cfg.F}, M={cfg.M}, L={cfg.L}, H1={cfg.H1}, H2={cfg.H2}, T={cfg.T}")
PY

PYTHONPATH="$REPO_ROOT/src" python "$EXPERIMENT_DIR/run_case.py" \
    "$EXPERIMENT_DIR/input.yaml" "$RUN_DIR" \
    > "$RUN_DIR/runner.out" 2> "$RUN_DIR/runner.err"

