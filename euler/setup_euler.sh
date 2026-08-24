#!/usr/bin/bash
# One-time environment setup for the bound-test harness AND the study harness
# on ETH Euler.
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
# run_studies.py is optional (the bound tests work without it) but if it is there
# it MUST sit next to test.py, which it imports by path.
HAVE_STUDIES=0
[ -f "$PROJECT/run_studies.py" ] && HAVE_STUDIES=1

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
# SciPy is optional: run_studies uses linear_sum_assignment for the
# symmetry-corrected first-period flip metric, and falls back to brute force for
# F <= 8 (and reports NaN above that) when it is absent.
python -m pip install scipy || echo "note: scipy not installed; flip_matched will"\
    " fall back to brute force for F <= 8"
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
echo "== smoke test 2: can the bound harness import the project? =="
cd "$PROJECT"
python test.py --tests analytic --no-plots --out "$PROJECT/results" \
               --name euler_check | tail -20

echo
echo "== smoke test 2b: are BOTH MILP encodings present and equivalent? =="
# rainflow_v2 carries the 'indicator' (original) and 'bigm' encodings of the same
# model. They must reach the SAME optimum -- a disagreement means a big-M is
# wrong, and it is far cheaper to find that here than in a 20-shard array.
if python -c "import fleet_management.degradation_model.rainflow_v2" 2>/dev/null; then
    python - <<'FORMCHECK'
from fleet_management.config import load_config
from fleet_management.degradation_model import rainflow_v2 as rf

p, b = 0.3, 0.12
data = {"model": "rainflow", "bound_method": "cantelli", "repair_model": "ard1",
        "F": 3, "M": 1, "L": 1, "H": 3,
        "tau": 1.0, "epsilon": 0.1, "rho": 0.8, "mu_0": 0.05, "v_0": 0.0,
        "mu": p * b, "v": p * (1 - p) * b * b, "support": b, "cgf": 0.1,
        "C_M": 1.0, "C_R": 0.5, "C_S": 2.0, "C_P": 1.0}
out = {}
for form in ("indicator", "bigm"):
    r = rf.solve(load_config(data), verbose=0, mip_gap=1e-9, time_limit=60,
                 reliability_impl="tangent", formulation=form)
    md = r["model"]
    out[form] = r["objective"]
    print(f"  {form:<10} obj={r['objective']:.6f}  binaries={md.NumBinVars}  "
          f"rows={md.NumConstrs}  genconstrs={md.NumGenConstrs}")
    md.dispose()
a, c = out["indicator"], out["bigm"]
assert abs(a - c) <= 1e-6 * max(1.0, abs(a)), (
    f"the two encodings disagree ({a} vs {c}) -- do not submit jobs")
print("  both encodings agree; FORM=indicator|bigm is safe to use")
FORMCHECK
else
    echo "  WARNING rainflow_v2.py not found in the package -- FORM=bigm will fail."
    echo "          Put it in fleet_management/degradation_model/ next to base.py."
fi

if [ "$HAVE_STUDIES" = "1" ]; then
    echo
    echo "== smoke test 3: study harness plan (no solving, no licence needed) =="
    # --plan exercises the ladders, the size guard and the per-bound feasibility
    # screen. A 'no:<bound>' in the screen column means that rung will return
    # 'infeasible' in milliseconds instead of a solve-time measurement.
    python run_studies.py --plan --studies scaling,horizon,heatmap,convergence \
        | grep -E "^base case|^combos|^  -> |^TOTAL|^Worst case|^WARNING|no:" || true

    echo
    echo "== smoke test 4: study harness dry run (validates every input) =="
    python run_studies.py --studies scaling --dry-run --no-plots \
        --factors 1 --out "$PROJECT/results" --name euler_check 2>&1 | tail -5

    echo
    echo "== smoke test 5: does the study harness accept --formulation? =="
    python run_studies.py --plan --studies scaling --formulation bigm \
        --factors 1 2>&1 | grep -E "^base case|^TOTAL|error|unrecognized" || true
fi

echo
echo "Setup done. Submit work with:"
echo "  bash euler/submit.sh sweep 20                    # bound tests (test.py)"
echo "  FORM=bigm bash euler/submit.sh sweep 20          # ... with the big-M encoding"
echo "  bash euler/submit.sh formulation 4               # compare both encodings"
if [ "$HAVE_STUDIES" = "1" ]; then
    echo "  bash euler/submit_studies.sh scaling 12          # studies (run_studies.py)"
    echo "  FORM=bigm bash euler/submit_studies.sh scaling 12  # ... big-M (lp_gap!)"
fi
