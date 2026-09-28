#!/usr/bin/bash
#SBATCH --job-name=gamma-H1-H2-grid
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=1G
#SBATCH --output=runs/gamma-H1-H2-grid-%j.out
#SBATCH --error=runs/gamma-H1-H2-grid-%j.err

set -Eeuo pipefail

REPO_ROOT=${SLURM_SUBMIT_DIR:-$PWD}
EXPERIMENT_REL=experiments/gamma_horizon_grid_final
EXPERIMENT_DIR="$REPO_ROOT/$EXPERIMENT_REL"
RUNS_DIR="$EXPERIMENT_DIR/runs"
JOB_ID=${SLURM_JOB_ID:-manual}
RUN_NAME="gamma_horizon_grid_final_$JOB_ID"
RUN_DIR="$RUNS_DIR/$RUN_NAME"
ARCHIVE="$RUNS_DIR/gamma_horizon_grid_final_${JOB_ID}.tar.gz"
INPUT_NAME=horizon_grid.yaml
REPORT="$RUN_DIR/horizon_grid_report.yaml"

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

cp "$EXPERIMENT_DIR/gamma_horizon_grid_final.sbatch" "$RUN_DIR/job.sbatch"
cp "$EXPERIMENT_DIR/README.md" "$RUN_DIR/README.md"
cp "$EXPERIMENT_DIR/summarize_grid.py" "$RUN_DIR/summarize_grid.py"
cp "$EXPERIMENT_DIR/plot_grid.py" "$RUN_DIR/plot_grid.py"
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
    echo "experiment=gamma_horizon_grid_final"
    echo "job_id=$JOB_ID"
    echo "host=$(hostname)"
    echo "started_at=$(date --iso-8601=seconds)"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
    echo "git_branch=$(git branch --show-current 2>/dev/null || echo unavailable)"
    echo "slurm_cpus_per_task=${SLURM_CPUS_PER_TASK:-unset}"
    echo "gurobi_threads=32"
    echo "H1_candidates=2,4,6"
    echo "H2_candidates=8,12,16,24"
    echo "evaluation_horizon=52"
    echo "per_case_time_limit_seconds=3300"
    echo "phase_I_time_limit_seconds=300"
    echo "solver_mip_gap=0.05"
    echo "reliability=tangent_cantelli"
    echo "tau=1.0"
    echo "epsilon=0.0001"
    echo "initial_fleet=non_pristine"
    echo "replacement=disabled"
} > "$RUN_DIR/metadata.txt"

module list > "$RUN_DIR/modules.txt" 2>&1 || true
"$PYTHON" -m pip freeze > "$RUN_DIR/python_packages.txt" 2>&1 || true

"$PYTHON" - "$INPUT" <<'PY' > "$RUN_DIR/preflight.txt" 2>&1
from pathlib import Path
import os
import sys

import gurobipy
import numpy as np
import yaml

import fleet_management
from fleet_management.config import load_config

path = Path(sys.argv[1])
raw = yaml.safe_load(path.read_text(encoding="utf-8"))
cfg = load_config(raw)
params = raw["gurobi_params"]
allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", "32"))

assert (cfg.F, cfg.M, cfg.L) == (4, 2, 2)
assert (cfg.H1, cfg.H2, cfg.T) == (2, 8, 10)
assert cfg.models == ["gamma", "rainflow"]
assert np.allclose(cfg.tau, 1.0)
assert np.allclose(cfg.epsilon, 1e-4)
assert np.all(cfg.mu_0 > 0.0)
assert raw["reliability_impl"] == "tangent"
assert raw["allow_replacement"] is False
assert float(raw["mip_gap"]) == 0.05
assert int(raw["time_limit"]) == 3300
assert raw["relaxation_warm_start"] is True
assert int(raw["relaxation_time_limit"]) == 300
assert params == {
    "Threads": 32,
    "Seed": 1,
    "MIPFocus": 3,
    "Heuristics": 0.02,
    "Symmetry": 2,
    "Cuts": 2,
}
assert allocated == params["Threads"]

print(f"PASS {path}")
print(f"F={cfg.F}, M={cfg.M}, L={cfg.L}, placeholder H1={cfg.H1}, H2={cfg.H2}")
print("Models:", cfg.models)
print("Initial means:", cfg.mu_0.tolist())
print("Initial variances:", cfg.v_0.tolist())
print("Slurm CPUs / Gurobi threads:", allocated, "/", params["Threads"])
print("Gurobi:", gurobipy.gurobi.version())
print("fleet_management:", fleet_management.__file__)
PY

cat "$RUN_DIR/preflight.txt"

/usr/bin/time -v "$PYTHON" "$RUN_DIR/run_horizon_sweep.py" \
    "$RUN_DIR/$INPUT_NAME" \
    "$REPORT" \
    --h1 2 4 6 \
    --h2 8 12 16 24 \
    --warm-start \
    --evaluation-horizon 52 \
    > "$RUN_DIR/runner.out" \
    2> "$RUN_DIR/runner.err"

"$PYTHON" "$RUN_DIR/summarize_grid.py" "$REPORT" "$RUN_DIR" \
    > "$RUN_DIR/postprocess.out" 2> "$RUN_DIR/postprocess.err"

MPLBACKEND=Agg "$PYTHON" "$RUN_DIR/plot_grid.py" \
    "$REPORT" "$RUN_DIR/horizon_grid.png" \
    >> "$RUN_DIR/postprocess.out" 2>> "$RUN_DIR/postprocess.err"

cat "$RUN_DIR/postprocess.out"
