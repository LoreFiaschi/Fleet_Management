#!/usr/bin/bash
#SBATCH --job-name=vbz_year_merge
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4g
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Reduce step: concatenates the per-strategy result rows and the per-strategy
# checkpoint traces, then ranks the strategies. Solves nothing, so it loads NO
# gurobi module and holds no licence token.
set -euo pipefail

PROJECT=${PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}
RUN_STAMP=${RUN_STAMP:?RUN_STAMP must be exported by euler/submit_year.sh}
TEST=${TEST:-year}
OUT=${OUT:-$PROJECT/results}

if ! command -v module >/dev/null 2>&1; then
    for init in /etc/profile.d/modules.sh \
                /cluster/apps/modules/init/bash \
                /usr/share/lmod/lmod/init/bash; do
        [ -f "$init" ] && . "$init" && break
    done
fi
set +e
module purge
module load stack/2024-06 python/3.12.8
set -e
source "$PROJECT/.venv-euler/bin/activate"
cd "$PROJECT"

python - "$OUT/${RUN_STAMP}_${TEST}" <<'PY'
import csv, pathlib, sys

d = pathlib.Path(sys.argv[1])
files = [f for f in sorted(d.glob("results_*.csv")) if f.name != "merged_results.csv"]
rows = []
for f in files:
    rows += list(csv.DictReader(open(f)))
if not rows:
    print(f"nothing merged in {d}; check the array logs")
    raise SystemExit(1)

cols = list(rows[0])
with open(d / "merged_results.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)

# checkpoint traces, tagged by strategy: one tidy file for the progress plots
ck = [f for f in sorted(d.glob("checkpoints_*.csv"))
      if f.name != "merged_checkpoints.csv"]
if ck:
    out, hdr = [], None
    for f in ck:
        strat = f.stem.replace("checkpoints_", "")
        for r in csv.DictReader(open(f)):
            hdr = hdr or (["strategy"] + list(r))
            out.append({"strategy": strat, **r})
    with open(d / "merged_checkpoints.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=hdr)
        w.writeheader()
        w.writerows(out)

def sortkey(r):
    return (r["objective"] == "", float(r["objective"] or "inf"))

rows.sort(key=sortkey)
L = [f"run folder  : {d}",
     f"result files: {len(files)}   checkpoint files: {len(ck)}", ""]
L.append(f"{'strategy':<24}{'status':<12}{'objective':>14}{'gap':>9}"
         f"{'bound':>13}{'sols':>6}{'nodes':>10}{'wall_s':>9}")
for r in rows:
    obj = f"{float(r['objective']):.3f}" if r["objective"] else "-"
    gap = f"{float(r['mip_gap']):.3f}" if r["mip_gap"] else "-"
    L.append(f"{r['value']:<24}{r['status']:<12}{obj:>14}{gap:>9}"
             f"{float(r['obj_bound'] or 'nan'):>13.1f}"
             f"{float(r['sol_count'] or 0):>6.0f}{float(r['nodes'] or 0):>10.0f}"
             f"{float(r['wall_s']):>9.0f}")
best = rows[0]
L += ["", f"best incumbent: {best['value']} -> {best['objective'] or 'NONE'}"]
if best["objective"] == "":
    L.append("NO strategy produced an incumbent. The model is feasible "
             "(feas_oracle.py exhibits a schedule), so this is a solver "
             "problem: raise SOLVE_TL, or check the damage-cap flag below.")
caps = {r["value"]: r.get("feasible_hint") for r in rows}
if any(v == "cap_on" for v in caps.values()):
    L.append(f"WARNING aggregate damage cap present: {caps} -- base.py/config.py "
             f"are not patched (derivation.md section 8)")
missing = [r["value"] for r in rows if r["status"] == "error"]
if missing:
    L.append(f"WARNING strategies that errored: {missing} (see traceback column)")
txt = "\n".join(L)
(d / "merged_summary.txt").write_text(txt + "\n")
print(txt)
PY
