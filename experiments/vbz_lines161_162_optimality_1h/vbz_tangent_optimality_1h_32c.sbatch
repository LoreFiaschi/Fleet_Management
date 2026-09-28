#!/usr/bin/bash
#SBATCH --job-name=vbz-tan-opt-32c
#SBATCH --time=01:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --output=runs/vbz-tan-opt-32c-%j.out
#SBATCH --error=runs/vbz-tan-opt-32c-%j.err

set -Eeuo pipefail

REPO_ROOT=${SLURM_SUBMIT_DIR:-$PWD}
EXPERIMENT_REL=experiments/vbz_lines161_162_optimality_1h
EXPERIMENT_DIR="$REPO_ROOT/$EXPERIMENT_REL"
RUNS_DIR="$EXPERIMENT_DIR/runs"
JOB_ID=${SLURM_JOB_ID:-manual}
RUN_NAME="vbz_tangent_optimality_1h_32c_$JOB_ID"
RUN_DIR="$RUNS_DIR/$RUN_NAME"
ARCHIVE="$RUNS_DIR/vbz_tangent_optimality_1h_32c_${JOB_ID}.tar.gz"
INPUT_NAME=vbz_lines161_162_F4_L2_tangent_boundfocus_1h_32c.yaml

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
            | sort -z \
            | xargs -0 -r sha256sum > SHA256SUMS
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

if [[ ! -f pyproject.toml || ! -d "$EXPERIMENT_DIR/input" ]]; then
    echo "Submit from the Fleet_Management repository root." >&2
    exit 2
fi

PYTHON="$REPO_ROOT/.venv/bin/python"

cp "$EXPERIMENT_DIR/vbz_tangent_optimality_1h_32c.sbatch" "$RUN_DIR/job.sbatch"
cp "$EXPERIMENT_DIR/run_case.py" "$RUN_DIR/run_case.py"
cp "$EXPERIMENT_DIR/README.md" "$RUN_DIR/README.md"
cp "$EXPERIMENT_DIR/input/$INPUT_NAME" "$RUN_DIR/$INPUT_NAME"
cp pyproject.toml "$RUN_DIR/source_snapshot/pyproject.toml"
cp src/fleet_management/config.py "$RUN_DIR/source_snapshot/config.py"
cp src/fleet_management/solver.py "$RUN_DIR/source_snapshot/solver.py"
cp src/fleet_management/degradation_model/base.py "$RUN_DIR/source_snapshot/base.py"
cp src/fleet_management/degradation_model/gamma.py "$RUN_DIR/source_snapshot/gamma.py"
cp src/fleet_management/degradation_model/rainflow.py "$RUN_DIR/source_snapshot/rainflow.py"

{
    echo "experiment=vbz_lines161_162_tangent_optimality_1h_32c"
    echo "job_id=$JOB_ID"
    echo "host=$(hostname)"
    echo "started_at=$(date --iso-8601=seconds)"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
    echo "git_status_begin"
    git status --porcelain --untracked-files=all 2>/dev/null || true
    echo "git_status_end"
    echo "solver_time_limit_seconds=3600"
    echo "solver_mip_gap=0.05"
    echo "reliability_impl=tangent"
    echo "tangent_ref=0.32"
    echo "slurm_cpus_per_task=${SLURM_CPUS_PER_TASK:-unset}"
    echo "gurobi_threads=32"
    echo "mip_focus=3"
    echo "heuristics=0.02"
    echo "symmetry=2"
    echo "cuts=2"
} > "$RUN_DIR/metadata.txt"

module list > "$RUN_DIR/modules.txt" 2>&1 || true
"$PYTHON" -m pip freeze > "$RUN_DIR/python_packages.txt" 2>&1 || true

"$PYTHON" - "$RUN_DIR/$INPUT_NAME" <<'PY' > "$RUN_DIR/preflight.txt" 2>&1
from pathlib import Path
import os
import sys
import yaml
import gurobipy
import fleet_management
from fleet_management.config import load_config

path = Path(sys.argv[1])
raw = yaml.safe_load(path.read_text(encoding="utf-8"))
cfg = load_config(raw)
params = raw["gurobi_params"]
allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", "32"))

assert (cfg.F, cfg.M, cfg.L) == (4, 2, 2)
assert (cfg.H1, cfg.H2, cfg.T) == (4, 12, 16)
assert raw["reliability_impl"] == "tangent"
assert float(raw["tangent_ref"]) == 0.32
assert float(raw["mip_gap"]) == 0.05
assert int(raw["time_limit"]) == 3600
assert raw["allow_replacement"] is False
assert params == {
    "Threads": 32,
    "Seed": 1,
    "MIPFocus": 3,
    "Heuristics": 0.02,
    "Symmetry": 2,
    "Cuts": 2,
}
assert allocated == params["Threads"], (
    f"Slurm allocated {allocated} CPUs but Gurobi requests {params['Threads']} threads"
)

print(f"PASS: F={cfg.F}, M={cfg.M}, L={cfg.L}, H1={cfg.H1}, H2={cfg.H2}, T={cfg.T}")
print("Reliability: tangent Cantelli at reference", raw["tangent_ref"])
print("Slurm CPUs / Gurobi threads:", allocated, "/", params["Threads"])
print("Gurobi parameters:", params)
print("Gurobi:", gurobipy.gurobi.version())
print("fleet_management:", fleet_management.__file__)
PY

cat "$RUN_DIR/preflight.txt"

"$PYTHON" "$RUN_DIR/run_case.py" \
    "$RUN_DIR/$INPUT_NAME" \
    "$RUN_DIR/result_lines161_162.yaml" \
    "$RUN_DIR/summary.yaml" \
    > "$RUN_DIR/runner.out" \
    2> "$RUN_DIR/runner.err"

cat "$RUN_DIR/summary.yaml"
