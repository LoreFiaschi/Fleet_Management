#!/usr/bin/bash
#SBATCH --job-name=vbz-161-162-F4
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --output=runs/vbz-161-162-F4-%j.out
#SBATCH --error=runs/vbz-161-162-F4-%j.err

set -Eeuo pipefail

REPO_ROOT=${SLURM_SUBMIT_DIR:-$PWD}
EXPERIMENT_REL=experiments/vbz_lines161_162_5pct
EXPERIMENT_DIR="$REPO_ROOT/$EXPERIMENT_REL"
RUNS_DIR="$EXPERIMENT_DIR/runs"
JOB_ID=${SLURM_JOB_ID:-manual}
RUN_NAME="lines161_162_F4_$JOB_ID"
RUN_DIR="$RUNS_DIR/$RUN_NAME"
ARCHIVE="$RUNS_DIR/vbz_lines161_162_F4_5pct_${JOB_ID}.tar.gz"

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

cp "$EXPERIMENT_DIR/vbz_lines161_162_5pct.sbatch" "$RUN_DIR/job.sbatch"
cp "$EXPERIMENT_DIR/run_case.py" "$RUN_DIR/run_case.py"
cp "$EXPERIMENT_DIR/summarize_cases.py" "$RUN_DIR/summarize_cases.py"
cp "$EXPERIMENT_DIR/README.md" "$RUN_DIR/README.md"
cp "$EXPERIMENT_DIR/input/"*.yaml "$RUN_DIR/"
cp pyproject.toml "$RUN_DIR/source_snapshot/pyproject.toml"
cp src/fleet_management/config.py "$RUN_DIR/source_snapshot/config.py"
cp src/fleet_management/solver.py "$RUN_DIR/source_snapshot/solver.py"
cp src/fleet_management/degradation_model/base.py "$RUN_DIR/source_snapshot/base.py"
cp src/fleet_management/degradation_model/gamma.py "$RUN_DIR/source_snapshot/gamma.py"
cp src/fleet_management/degradation_model/rainflow.py "$RUN_DIR/source_snapshot/rainflow.py"

{
    echo "experiment=vbz_lines161_162_F4_5pct"
    echo "job_id=$JOB_ID"
    echo "host=$(hostname)"
    echo "started_at=$(date --iso-8601=seconds)"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
    echo "git_status_begin"
    git status --porcelain --untracked-files=all 2>/dev/null || true
    echo "git_status_end"
    echo "solver_time_limit_seconds=55800"
    echo "solver_mip_gap=0.05"
    echo "case=lines161_162"
    echo "threads=4"
    echo "mip_focus=3"
} > "$RUN_DIR/metadata.txt"

module list > "$RUN_DIR/modules.txt" 2>&1 || true
"$PYTHON" -m pip freeze > "$RUN_DIR/python_packages.txt" 2>&1 || true

"$PYTHON" - "$RUN_DIR" <<'PY' > "$RUN_DIR/preflight.txt" 2>&1
from pathlib import Path
import sys
import yaml
import gurobipy
import fleet_management
from fleet_management.config import load_config

run_dir = Path(sys.argv[1])
expected = {"vbz_lines161_162_F4_H2_12_5pct.yaml": (4, 2, 4)}
for name, dimensions in expected.items():
    path = run_dir / name
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg = load_config(raw)
    assert (cfg.F, cfg.M, cfg.L) == dimensions
    assert (cfg.H1, cfg.H2, cfg.T) == (4, 12, 16)
    assert float(raw["mip_gap"]) == 0.05
    assert int(raw["time_limit"]) == 55800
    assert raw["allow_replacement"] is False
    assert int(raw["gurobi_params"]["Threads"]) == 4
    print(f"PASS {name}: F={cfg.F}, M={cfg.M}, L={cfg.L}, T={cfg.T}")
print("Gurobi:", gurobipy.gurobi.version())
print("fleet_management:", fleet_management.__file__)
PY

"$PYTHON" "$RUN_DIR/run_case.py" \
    lines161_162 \
    "$RUN_DIR/vbz_lines161_162_F4_H2_12_5pct.yaml" \
    "$RUN_DIR/result_lines161_162.yaml" \
    "$RUN_DIR/case_lines161_162_summary.json" \
    > "$RUN_DIR/runner_lines161_162.out" \
    2> "$RUN_DIR/runner_lines161_162.err"

"$PYTHON" "$RUN_DIR/summarize_cases.py" "$RUN_DIR" \
    > "$RUN_DIR/postprocess.out" 2> "$RUN_DIR/postprocess.err"

cat "$RUN_DIR/postprocess.out"
