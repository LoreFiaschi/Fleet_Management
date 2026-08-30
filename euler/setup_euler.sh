#!/usr/bin/bash
# One-time environment setup for the bound-test harness AND the study harness
# on ETH Euler.
#   cd ~/<your-project> && bash euler/setup_euler.sh
#
# Creates .venv-euler on top of the Euler python module. The module already ships
# numpy / matplotlib / pyyaml, so --system-site-packages means the only things
# that have to come from PyPI are gurobipy and scipy.
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
# SciPy is REQUIRED, not optional, since the sparse assembly landed:
#   * rainflow_sparse builds the constraint matrix as COO triplets and converts
#     to CSR before handing it to addMConstr, so formulation='sparse' /
#     'bigm_sparse' raise ImportError without it;
#   * run_studies uses linear_sum_assignment for the symmetry-corrected
#     first-period flip metric (it can fall back to brute force for F <= 8).
python -m pip install scipy
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
echo "== smoke test 2b: are ALL FOUR formulations present and equivalent? =="
# `formulation` flattens a 2x2 grid of (encoding, assembly):
#
#                        assembly='loop'      assembly='sparse'
#   encoding='indicator'   'indicator'          'sparse'
#   encoding='bigm'        'bigm'               'bigm_sparse'
#
# The ENCODING is a modelling choice: indicator and bigm describe the same
# integer feasible set but different LP relaxations, so their OPTIMA must agree
# -- a disagreement means a big-M is wrong, and it is far cheaper to find that
# here than in a 20-shard array. The ASSEMBLY is not a modelling choice at all:
# for a fixed encoding the two must produce the identical model, so their sizes
# must agree too, exactly and not approximately.
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
obj, size = {}, {}
for form in ("indicator", "sparse", "bigm", "bigm_sparse"):
    r = rf.solve(load_config(data), verbose=0, mip_gap=1e-9, time_limit=60,
                 reliability_impl="tangent", formulation=form)
    md = r["model"]
    obj[form] = r["objective"]
    size[form] = (md.NumBinVars, md.NumConstrs, md.NumGenConstrs, md.NumNZs)
    print(f"  {form:<12} obj={r['objective']:.6f}  binaries={md.NumBinVars}  "
          f"rows={md.NumConstrs}  genconstrs={md.NumGenConstrs}  "
          f"nnz={md.NumNZs}")
    md.dispose()

# 1. the ENCODING axis: same optimum, different model
a, c = obj["indicator"], obj["bigm"]
assert abs(a - c) <= 1e-6 * max(1.0, abs(a)), (
    f"the two encodings disagree ({a} vs {c}) -- do not submit jobs")

# 2. the ASSEMBLY axis: the SAME model, so the sizes must match exactly. This is
#    a weak check compared with test_sparse_version.py's row-by-row comparison,
#    but it costs nothing and catches a half-installed rainflow_sparse.
for enc, sp in (("indicator", "sparse"), ("bigm", "bigm_sparse")):
    assert size[enc] == size[sp], (
        f"{enc} and {sp} must build the IDENTICAL model, but their sizes "
        f"differ: {size[enc]} vs {size[sp]}. Run "
        f"'python test_sparse_version.py --tests equivalence' for the "
        f"row-by-row diff before submitting anything.")
    assert abs(obj[enc] - obj[sp]) <= 1e-9 * max(1.0, abs(obj[enc])), (
        f"{enc} and {sp} disagree on the optimum ({obj[enc]} vs {obj[sp]})")

# 3. the loop-closure rows must be there at all. Without rainflow_v2's
#    'repeatability' hook the operating phase closes on the MEAN only, so a
#    cantelli cell can end the horizon with a larger variance than it started
#    with -- a "repeatable" cycle that is not repeatable in the quantity its own
#    reliability row reads. Silent, and it changes every result.
from fleet_management.degradation_model.rainflow_v2 import RainflowCellBuilder
assert getattr(RainflowCellBuilder, "repeatability", None) is not None, (
    "rainflow_v2.RainflowCellBuilder has no 'repeatability' hook: the loop "
    "closes on the mean only and every descriptor (v / R / K) is unconstrained "
    "at the end of the horizon. Update rainflow_v2.py before submitting.")
print("  all four formulations agree; "
      "FORM=indicator|bigm|sparse|bigm_sparse is safe to use")
print("  loop-closure hook present (v / R / K are closed, not just the mean)")
FORMCHECK
else
    echo "  WARNING rainflow_v2.py not found in the package -- FORM=bigm will fail."
    echo "          Put it in the package next to base.py -- that is"
    echo "          src/fleet_management/degradation_model/ in a src layout."
fi

echo
echo "== smoke test 2c: does the sparse-assembly harness run? =="
# One CI-sized pass of test_sparse_version.py: it builds every (encoding,
# assembly) pair and compares them as objects -- columns with their types,
# bounds and objective coefficients, and the linear / indicator / quadratic rows
# as canonicalised multisets. Seconds, and it is the check that actually has
# teeth. A non-zero exit means the two assemblies differ; do not submit.
if [ -f "$PROJECT/test_sparse_version.py" ]; then
    if python test_sparse_version.py --quick --no-plots \
            --out "$PROJECT/results" --name euler_check 2>&1 | tail -25; then
        echo "  sparse assembly verified; submit_sparse.sh is safe to use"
    else
        echo "  ERROR test_sparse_version.py failed -- the sparse assembly does" >&2
        echo "        NOT reproduce the reference build. Do not submit jobs" >&2
        echo "        with FORM=sparse or FORM=bigm_sparse." >&2
        exit 1
    fi
else
    echo "  note: test_sparse_version.py not found in the project root; skipping."
    echo "        (Needed only if you intend to use the sparse assembly.)"
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
echo "  bash euler/submit_sparse.sh                      # sparse assembly: equivalence + build cost"
echo "  EXCLUSIVE=1 bash euler/submit_sparse.sh          # ... with quotable timings"
if [ "$HAVE_STUDIES" = "1" ]; then
    echo "  bash euler/submit_studies.sh scaling 12          # studies (run_studies.py)"
    echo "  FORM=bigm bash euler/submit_studies.sh scaling 12  # ... big-M (lp_gap!)"
fi
