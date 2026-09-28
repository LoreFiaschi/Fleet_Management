"""Convert a long-format schedule CSV into the dict/file that plotter.py reads.

plotter.plot_management() wants a solver-output mapping with

    F, M, L                 sizes
    H1, H2                  horizon split (for the transitory | operating divider)
    tau (or alpha)          scalar, (L,), (F,), or (F, L)
    mu_0                    (F, L)          -> grid column 0
    mu                      (F, L, T)       -> grid columns 1..T
    x                       (F, M+1, T)     -> cell glyphs

while the CSV is one row per (vehicle, component, step):

    vehicle,component,step,activity,mission,repair,replace,mu,v

`mu_0` and `tau` do not appear in the CSV, so pass the solver *input* file with
--config to recover them; otherwise they fall back to 0.0 and 1.0.

Usage
-----
    python csv_to_plotter.py schedule.csv --config fleet.yaml -o sol.json
    python csv_to_plotter.py schedule.csv --config fleet.yaml -o sol.json --plot sched.png

Glyph convention (--gear):
    depot   (default) x[i,0,k]=1 on every depot step, matching the solver's
            assignment variable, so every depot step draws a gear.
    repair  x[i,0,k]=1 only where repair or replace fires; other depot steps are
            left unassigned and draw the idle "zzz" cloud.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

DEPOT_ACTIVITIES = {"depot", "maintenance", "maint", "0"}
IDLE_ACTIVITIES = {"idle", "none", "rest", "park", "parked", ""}


# ---------------------------------------------------------------------------
# CSV -> plotter dict
# ---------------------------------------------------------------------------
def convert(csv_path, config_path=None, gear="depot", h1=None, M=None):
    rows = _read_rows(csv_path)

    F = max(r["vehicle"] for r in rows) + 1
    L = max(r["component"] for r in rows) + 1
    T = max(r["step"] for r in rows) + 1
    M_csv = max(r["mission"] for r in rows)
    M = int(M) if M is not None else M_csv

    expected = F * L * T
    if len(rows) != expected:
        raise ValueError(
            f"CSV has {len(rows)} rows but F*L*T = {F}*{L}*{T} = {expected}; "
            "the (vehicle, component, step) grid is incomplete or duplicated."
        )
    if M_csv > M:
        raise ValueError(f"CSV uses mission index {M_csv} but M = {M}.")

    mu = np.full((F, L, T), np.nan)
    v = np.full((F, L, T), np.nan)
    x = np.zeros((F, M + 1, T))
    m_rep = np.zeros((F, L, T))
    r_rep = np.zeros((F, L, T))
    activity = np.empty((F, T), dtype=object)

    for r in rows:
        i, l, k = r["vehicle"], r["component"], r["step"]
        if not np.isnan(mu[i, l, k]):
            raise ValueError(f"duplicate row for vehicle={i}, component={l}, step={k}.")
        mu[i, l, k] = r["mu"]
        v[i, l, k] = r["v"]
        m_rep[i, l, k] = r["repair"]
        r_rep[i, l, k] = r["replace"]

        # activity / mission live on the vehicle, not the component: every
        # component row of the same (vehicle, step) must agree.
        prev = activity[i, k]
        if prev is None:
            activity[i, k] = (r["activity"], r["mission"])
        elif prev != (r["activity"], r["mission"]):
            raise ValueError(
                f"vehicle={i}, step={k}: components disagree on the activity "
                f"({prev} vs {(r['activity'], r['mission'])})."
            )

    for i in range(F):
        for k in range(T):
            act, j = activity[i, k]
            act_l = str(act).strip().lower()
            if j > 0:
                if act_l in DEPOT_ACTIVITIES | IDLE_ACTIVITIES:
                    raise ValueError(
                        f"vehicle={i}, step={k}: activity {act!r} contradicts "
                        f"mission index {j}."
                    )
                x[i, j, k] = 1.0
            elif act_l in DEPOT_ACTIVITIES:
                repaired = bool(m_rep[i, :, k].any() or r_rep[i, :, k].any())
                if gear == "depot" or repaired:
                    x[i, 0, k] = 1.0        # else: leave idle -> "zzz" cloud
            elif act_l in IDLE_ACTIVITIES:
                pass                        # unassigned -> "zzz" cloud
            else:
                raise ValueError(
                    f"vehicle={i}, step={k}: unrecognised activity {act!r} with "
                    f"mission 0; expected one of "
                    f"{sorted(DEPOT_ACTIVITIES | IDLE_ACTIVITIES)}."
                )

    mu_0, tau, H = _initial_state(config_path, F, L)
    H1 = int(h1) if h1 is not None else (int(H) if H is not None else T // 2)
    if not 0 < H1 <= T:
        raise ValueError(f"H1 = {H1} must satisfy 0 < H1 <= T = {T}.")

    return {
        "F": F, "M": M, "L": L, "T": T,
        "H": H1, "H1": H1, "H2": T - H1,
        "tau": tau.tolist(),
        "mu_0": mu_0.tolist(),
        "mu": mu.tolist(),
        "x": x.tolist(),
        # not read by the plotter, kept so the file round-trips the schedule
        "v": v.tolist(), "m": m_rep.tolist(), "r": r_rep.tolist(),
    }


def _read_rows(csv_path):
    required = {"vehicle", "component", "step", "activity", "mission",
                "repair", "replace", "mu", "v"}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing column(s): {sorted(missing)}.")
        out = []
        for row in reader:
            out.append({
                "vehicle": int(row["vehicle"]), "component": int(row["component"]),
                "step": int(row["step"]), "activity": row["activity"],
                "mission": int(float(row["mission"])),
                "repair": int(round(float(row["repair"]))),
                "replace": int(round(float(row["replace"]))),
                "mu": float(row["mu"]), "v": float(row["v"]),
            })
    return out


def _initial_state(config_path, F, L):
    """Recover mu_0, tau, and H from the solver input file (if given)."""
    if config_path is None:
        print("[warn] no --config: using mu_0 = 0 and tau = 1 for the colour scale.")
        return np.zeros((F, L)), np.ones((F, L)), None

    p = Path(config_path)
    with open(p) as f:
        cfg = yaml.safe_load(f) if p.suffix.lower() in (".yaml", ".yml") else json.load(f)

    mu_0 = _fl(cfg.get("mu_0", 0.0), F, L, "mu_0")
    tau = _fl(cfg.get("tau", cfg.get("alpha", 1.0)), F, L, "tau")
    H = cfg.get("H")
    if isinstance(H, (list, tuple)):
        H = H[0]
    return mu_0, tau, H


def _fl(value, F, L, name):
    """scalar / (L,) / (F,L) -> (F, L), the same broadcasting the solver uses."""
    a = np.asarray(value, dtype=float)
    if a.ndim == 0:
        return np.full((F, L), float(a))
    if a.shape == (L,):
        return np.tile(a.reshape(1, L), (F, 1))
    if a.shape == (F, L):
        return a
    raise ValueError(f"'{name}' shape {a.shape} must be scalar, ({L},), or ({F},{L}).")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="long-format schedule CSV")
    ap.add_argument("-o", "--out", default=None,
                    help="output .json / .yaml for plotter.py (default: <csv>.json)")
    ap.add_argument("--config", default=None,
                    help="solver input file, to recover mu_0 / tau / H")
    ap.add_argument("--gear", choices=("depot", "repair"), default="depot",
                    help="which depot steps get the gear glyph (default: depot)")
    ap.add_argument("--h1", type=int, default=None,
                    help="override the transitory horizon length")
    ap.add_argument("-M", type=int, default=None,
                    help="override M (default: max mission index in the CSV)")
    ap.add_argument("--plot", default=None,
                    help="also render the schedule here via plotter.plot_management")
    args = ap.parse_args()

    data = convert(args.csv, config_path=args.config, gear=args.gear,
                   h1=args.h1, M=args.M)

    out = Path(args.out) if args.out else Path(args.csv).with_suffix(".json")
    with open(out, "w") as f:
        if out.suffix.lower() in (".yaml", ".yml"):
            yaml.safe_dump(data, f, default_flow_style=None, sort_keys=False)
        else:
            json.dump(data, f)
    print(f"wrote {out}  (F={data['F']}, M={data['M']}, L={data['L']}, "
          f"T={data['T']}, H1={data['H1']}, H2={data['H2']})")

    if args.plot:
        from  src.fleet_management.utils.plotter import plot_management
        plot_management(str(out), args.plot)
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
