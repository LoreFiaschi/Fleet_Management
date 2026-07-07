"""Schedule visualisation.

An F x 2H grid: each row is a train, each column a time step, each cell split
into L horizontal strips (one per component) coloured on a green-to-red
heatmap by mu/tau. Strip border style encodes the component's degradation
model. Cell annotations show the assigned mission number, a maintenance
(gear) and/or replacement (wrench) marker, or an idle ("zzz") marker.
Reference: spec/spec.tex, "Output Specification" -> "Plot Layout".
"""

import json
import os
from pathlib import Path

import h5py
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import yaml

SUPPORTED_INPUT_EXTENSIONS = {".yaml", ".yml", ".json", ".h5", ".hdf5"}
SUPPORTED_PLOT_EXTENSIONS = {".png", ".pdf"}

# Border style per degradation model (spec: solid/dashed/dotted/dash-dot/long-dash).
_BORDER_STYLES = {
    "gaussian": "solid",
    "inverse_gaussian": "dashed",
    "wiener": "dotted",
    "gamma": "dashdot",
    "rainflow": (0, (8, 2)),
}


def plot_management(input_file_path: str, plot_file_path: str = None) -> None:
    """Plot solver output as a colour-coded F x 2H schedule grid.

    Parameters
    ----------
    input_file_path : str
        Path to a solver output file (YAML, JSON, or HDF5); may hold a
        single-horizon or multi-horizon (H interval) result.
    plot_file_path : str, optional
        Output image path, or a filename *prefix* for multi-horizon output:
        one file per H, e.g. "results/schedule" ->
        "results/schedule_H5.png", "results/schedule_H6.png", ...
        Defaults to "output.png" (scalar H) or "output" (interval H).
    """
    input_file = Path(input_file_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file_path}")
    if input_file.suffix.lower() not in SUPPORTED_INPUT_EXTENSIONS:
        raise ValueError(
            f"Unsupported input file type '{input_file.suffix}'. "
            f"Supported types: {sorted(SUPPORTED_INPUT_EXTENSIONS)}"
        )

    data = _read_output(input_file)

    if _is_multi_horizon(data):
        prefix, ext = _resolve_plot_prefix(plot_file_path)
        for h_key, res in data.items():
            path = Path(f"{prefix}_H{h_key}{ext}")
            _check_plot_dir(path)
            _plot_single(res, path)
    else:
        path = _resolve_plot_path(plot_file_path)
        _check_plot_dir(path)
        _plot_single(data, path)


# ======================================================================
# Path handling
# ======================================================================

def _resolve_plot_path(plot_file_path) -> Path:
    if plot_file_path is None:
        return Path("output.png")
    p = Path(plot_file_path)
    if p.suffix == "":
        return p.with_suffix(".png")
    if p.suffix.lower() not in SUPPORTED_PLOT_EXTENSIONS:
        raise ValueError(
            f"Unsupported plot file type '{p.suffix}'. Supported types: "
            f"{sorted(SUPPORTED_PLOT_EXTENSIONS)}"
        )
    return p


def _resolve_plot_prefix(plot_file_path):
    if plot_file_path is None:
        return "output", ".png"
    p = Path(plot_file_path)
    ext = p.suffix.lower()
    if ext == "":
        return str(p), ".png"
    if ext not in SUPPORTED_PLOT_EXTENSIONS:
        raise ValueError(
            f"Unsupported plot file type '{ext}'. Supported types: "
            f"{sorted(SUPPORTED_PLOT_EXTENSIONS)}"
        )
    return str(p.with_suffix("")), ext


def _check_plot_dir(path: Path) -> None:
    plot_dir = path.parent
    if plot_dir != Path("") and not plot_dir.exists():
        raise FileNotFoundError(f"Plot directory does not exist: {plot_dir}")
    if plot_dir != Path("") and not os.access(plot_dir, os.W_OK):
        raise PermissionError(f"Plot directory is not writable: {plot_dir}")


# ======================================================================
# Reading solver output
# ======================================================================

def _is_multi_horizon(data: dict) -> bool:
    return "status" not in data


def _read_output(input_file: Path) -> dict:
    ext = input_file.suffix.lower()
    if ext in (".yaml", ".yml"):
        with open(input_file, "r") as f:
            return yaml.safe_load(f)
    if ext == ".json":
        with open(input_file, "r") as f:
            return json.load(f)
    return _read_hdf5_output(input_file)


def _read_hdf5_output(path: Path) -> dict:
    with h5py.File(path, "r") as f:
        if "metadata" in f:
            return _read_hdf5_single(f)
        result = {}
        for key in f.keys():
            h_val = int(key[1:]) if key.startswith("H") else key
            result[h_val] = _read_hdf5_single(f[key])
        return result


def _read_hdf5_single(group: "h5py.Group") -> dict:
    meta = group["metadata"]
    res = {
        "status": meta.attrs["status"],
        "objective": float(meta.attrs["objective"]) if "objective" in meta.attrs else None,
        "F": int(meta.attrs["F"]), "H": int(meta.attrs["H"]),
        "M": int(meta.attrs["M"]), "L": int(meta.attrs["L"]),
    }
    if "solution" in group:
        sol = group["solution"]
        res["x"] = sol["x"][()]
        res["x_m"] = sol["x_m"][()]
        res["x_r"] = sol["x_r"][()]
        res["mu"] = sol["mu"][()]
        res["v"] = sol["v"][()]
        res["u"] = float(sol["u"][()])
        res["z"] = sol["z"][()]
    else:
        for key in ("x", "x_m", "x_r", "mu", "v", "u", "z"):
            res[key] = None
    params = group.get("parameters")
    res["tau"] = params["tau"][()] if params is not None and "tau" in params else None
    if params is not None and "model" in params:
        model_raw = params["model"][()]
        res["model"] = [[m.decode() if isinstance(m, bytes) else m for m in row] for row in model_raw]
    else:
        res["model"] = None
    return res


# ======================================================================
# Drawing
# ======================================================================

def _plot_single(res: dict, path: Path) -> None:
    if res.get("status") != "optimal" or res.get("x") is None:
        raise ValueError(
            f"Cannot plot a non-optimal result (status='{res.get('status')}'); "
            "no schedule is available."
        )

    F, L, H, M = res["F"], res["L"], res["H"], res["M"]
    two_h = 2 * H
    tau = np.asarray(res["tau"], dtype=float)
    mu = np.asarray(res["mu"], dtype=float)
    x = np.asarray(res["x"], dtype=float)
    x_m = np.asarray(res["x_m"], dtype=float)
    x_r = np.asarray(res["x_r"], dtype=float)
    model = res["model"]

    cmap = mcolors.LinearSegmentedColormap.from_list("gr", ["green", "red"])
    fig, ax = plt.subplots(figsize=(max(two_h * 0.8, 6), max(F * 0.8, 4)))

    strip_h = 1.0 / L
    for i in range(F):
        for k in range(two_h):
            for l in range(L):
                tau_il = tau[i][l]
                frac = np.clip(mu[i, l, k] / tau_il, 0.0, 1.0) if tau_il else 0.0
                style = _BORDER_STYLES.get(model[i][l], "solid")
                rect = mpatches.Rectangle(
                    (k - 0.5, i - 0.5 + l * strip_h), 1.0, strip_h,
                    facecolor=cmap(frac), edgecolor="black", linewidth=0.7,
                    linestyle=style,
                )
                ax.add_patch(rect)

    ax.set_xlim(-0.5, two_h - 0.5)
    ax.set_ylim(F - 0.5, -0.5)

    if L > 1:
        for i in range(F):
            for l in range(1, L):
                y = i - 0.5 + l * strip_h
                ax.hlines(y, -0.5, two_h - 0.5, colors="gray", linewidths=0.3, linestyles="--")

    # Half-horizon separator at k = H (0-based boundary between column H-1 and H)
    ax.axvline(H - 0.5, color="black", linewidth=1.5, linestyle="--")

    # --- Cell annotations ---
    for i in range(F):
        for k in range(two_h):
            if x[i, 0, k] == 1:
                has_repair = bool(np.any(x_m[i, :, k] == 1))
                has_replace = bool(np.any(x_r[i, :, k] == 1))
                if has_repair and has_replace:
                    _draw_gear(ax, k - 0.18, i)
                    _draw_wrench(ax, k + 0.18, i)
                elif has_repair:
                    _draw_gear(ax, k, i)
                elif has_replace:
                    _draw_wrench(ax, k, i)
                # else: a maintenance day with no repair/replacement -- blank
            else:
                assigned_j = next((j for j in range(1, M + 1) if x[i, j, k] == 1), None)
                if assigned_j is not None:
                    ax.text(
                        k, i, str(assigned_j), ha="center", va="center",
                        fontsize=10, fontweight="bold", color="black", zorder=5,
                    )
                else:
                    _draw_sleep_cloud(ax, k, i)

    ax.set_xticks(range(two_h))
    ax.set_xticklabels(range(1, two_h + 1))
    ax.set_yticks(range(F))
    ax.set_yticklabels(range(1, F + 1))
    ax.set_xlabel("Time step k")
    ax.set_ylabel("Train i")

    for i in range(F + 1):
        ax.axhline(i - 0.5, color="black", linewidth=0.5)
    for k in range(two_h + 1):
        ax.axvline(k - 0.5, color="black", linewidth=0.3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Mean degradation (μ / τ)", shrink=0.8)

    used_models = sorted({model[i][l] for i in range(F) for l in range(L)})
    legend_handles = [
        mlines.Line2D([], [], color="black", linestyle=_BORDER_STYLES.get(m, "solid"),
                      label=m.replace("_", " "))
        for m in used_models
    ]
    if legend_handles:
        ax.legend(
            handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.15),
            ncol=min(len(legend_handles), 5), fontsize=7, frameon=False,
        )

    obj = res.get("objective")
    obj_str = f"{obj:.4g}" if obj is not None else "N/A"
    ax.set_title(f"F={F}, M={M}, H={H}, L={L}, objective={obj_str}")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _draw_gear(ax, cx, cy):
    """Maintenance-day-with-repair marker (gear)."""
    ax.text(cx, cy, "⚙", ha="center", va="center", fontsize=13, color="black", zorder=6)


def _draw_wrench(ax, cx, cy):
    """Replacement marker: a bold 'R' in a filled circle."""
    ax.text(
        cx, cy, "R", ha="center", va="center", fontsize=9, fontweight="bold",
        color="white", zorder=6,
        bbox=dict(boxstyle="circle,pad=0.15", facecolor="darkred", edgecolor="none"),
    )


def _draw_sleep_cloud(ax, cx, cy):
    """Idle-day marker: a comic cloud with 'zzz'."""
    cloud = mpatches.FancyBboxPatch(
        (cx - 0.3, cy - 0.2), 0.6, 0.4, boxstyle="round,pad=0.05",
        facecolor="white", edgecolor="gray", linewidth=0.8, alpha=0.85, zorder=4,
    )
    ax.add_patch(cloud)
    ax.text(cx, cy, "zzz", ha="center", va="center", fontsize=8, fontstyle="italic",
             color="darkblue", zorder=6)
