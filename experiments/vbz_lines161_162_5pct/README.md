# VBZ lines 161 and 162 inspired 5% experiment

This experiment runs a two-line showcase. A separate line 162 control job is
included only as a fallback and is not launched by the primary job. The
official ZVV line descriptions are:

- line 161: Zürich, Bürkliplatz to Kilchberg ZH, Neuweid;
- line 162: Kilchberg ZH, Asp to Kilchberg ZH, Bahnhof.

The routes share part of the Kilchberg corridor. The fleet sizes, degradation
increments and costs remain synthetic and must not be described as VBZ data.

## Cases

| Case | F | M | L | H1 | H2 | T | MIP-gap target |
|---|---:|---:|---:|---:|---:|---:|---:|
| Lines 161 and 162 | 3 | 2 | 4 | 4 | 12 | 16 | 5% |

Fallback, if needed: line 162 only with `F=2`, `M=1`, and otherwise identical
dimensions and solver settings.

One model period represents one aggregate week. The four-week transitory phase
followed by four repetitions of the 12-week operating cycle gives the annual
interpretation `4 + 4 * 12 = 52` weeks. The optimization model contains only
`T = H1 + H2 = 16` periods.

The primary job runs one Gurobi process with four threads. It may use up to 15
hours 30 minutes inside one 16-hour Slurm allocation. If the 5% target is not
reached, any feasible incumbent and its final bound remain reportable.

Replacement stays disabled to preserve comparability with the successful
20%-gap line 162 experiment and to keep the showcase tractable.

## Submit

From the Fleet_Management repository root on Euler:

```bash
sbatch experiments/vbz_lines161_162_5pct/vbz_lines161_162_5pct.sbatch
```

Do not submit the fallback simultaneously. If the showcase is not usable, run:

```bash
sbatch experiments/vbz_lines161_162_5pct/vbz_line162_5pct_fallback.sbatch
```

## Download

After job `JOBID` finishes:

```powershell
scp clangenauer@euler.ethz.ch:~/Fleet_Management/experiments/vbz_lines161_162_5pct/runs/vbz_lines161_162_5pct_JOBID.tar.gz .
scp clangenauer@euler.ethz.ch:~/Fleet_Management/experiments/vbz_lines161_162_5pct/runs/vbz_lines161_162_5pct_JOBID.tar.gz.sha256 .
```
