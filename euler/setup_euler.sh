#!/usr/bin/bash
# One-time environment setup for the bound-test harness on ETH Euler.
#   cd ~/<your-project> && bash euler/setup_euler.sh
#
# Creates .venv-euler on top of the Euler python module. The module already ships
# numpy / matplotlib / pyyaml, so --system-site-packages means the only thing that
# has to come from PyPI is gurobipy.
set -euo pipefail

# Project root is derived from THIS script's location, never hardcoded: the
# directory may be called Fleet_Management, Fleet_management or anything else,
# and Linux paths are case sensitive.
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$PROJECT/.venv-euler"

STACK="stack/2024-06"
PYTHON="python/3.12.8"
GUROBI="gurobi/12.0.1"          # module version and gurobipy version MUST match

echo "== project root: $PROJECT"
if git -C "$PROJECT" rev-parse --git-dir >/dev/null 2>&1; then
    echo "== branch      : $(git -C "$PROJECT" rev-parse --abbrev-ref HEAD)"
    echo "== commit      : $(git -C "$PROJECT" rev-parse --short HEAD)"
fi
if [ ! -f "$PROJECT/test.py" ]; then
    echo "ERROR: no test.py in $PROJECT -- upload the harness first (see euler.md §1)" >&2
    exit 1
fi

module purge
module load $STACK $PYTHON $GUROBI
module load eth_proxy || echo "note: eth_proxy not loaded; pip may not reach PyPI"

echo "== python: $(python --version 2>&1), $(which python)"
echo "== gurobi: $(gurobi_cl --version 2>/dev/null | head -1 || echo 'gurobi_cl not found')"

if [ ! -d "$VENV" ]; then
    python -m venv --system-site-packages "$VENV"
fi
source "$VENV/bin/activate"
python -m pip install --upgrade pip
python -m pip install "gurobipy==${GUROBI#gurobi/}"
if [ -f "$PROJECT/pyproject.toml" ]; then
    python -m pip install -e "$PROJECT"          # the fleet_management package
fi

echo
echo "== licence tokens (ETH academic licence, 4096 shared tokens) =="
gurobi_cl --tokens || echo "could not query tokens"

echo
echo "== smoke test 1: can gurobipy build and solve a tiny model? =="
python - <<'PY'
import gurobipy as gp
m = gp.Model(); m.Params.OutputFlag = 0
x = m.addVar(ub=3); m.setObjective(x, gp.GRB.MAXIMIZE); m.optimize()
print(f"gurobipy {gp.gurobi.version()} ok, objective={m.ObjVal}")
PY

echo
echo "== smoke test 2: can the harness import the project and validate inputs? =="
cd "$PROJECT"
python test.py --tests analytic --no-plots --out "$PROJECT/results" \
               --name euler_check | tail -20

echo
echo "Setup done. Submit work with:  bash euler/submit.sh sweep 20"
