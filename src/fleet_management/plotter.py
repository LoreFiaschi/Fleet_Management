import json
import os
from pathlib import Path

import h5py
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import yaml

SUPPORTED_INPUT_EXTENSIONS = {".yaml", ".yml", ".json", ".h5", ".hdf5"}
SUPPORTED_PLOT_EXTENSIONS = {".png", ".pdf"}


def plot_management(input_file_path: str, plot_file_path: str = None) -> None:
    """
    Plot the fleet management schedule as a coloured grid.

    Parameters
    ----------
    input_file_path : str
        Path to a solver output file (YAML, JSON, or HDF5).
    plot_file_path : str, optional
        Path where the plot will be saved. Defaults to "output.png".
        If provided without an extension, ".png" is appended.
    """
    # --- Consistency checks ---
    input_file = Path(input_file_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file_path}")
    if input_file.suffix.lower() not in SUPPORTED_INPUT_EXTENSIONS:
        raise ValueError(
            f"Unsupported input file type '{input_file.suffix}'. "
            f"Supported types: {sorted(SUPPORTED_INPUT_EXTENSIONS)}"
        )

    plot_path = _resolve_plot_path(plot_file_path)
    if plot_path.suffix.lower() not in SUPPORTED_PLOT_EXTENSIONS:
        raise ValueError(
            f"Unsupported plot file type '{plot_path.suffix}'. "
            f"Supported types: {sorted(SUPPORTED_PLOT_EXTENSIONS)}"
        )
    plot_dir = plot_path.parent
    if plot_dir != Path("") and not plot_dir.exists():
        raise FileNotFoundError(f"Plot directory does not exist: {plot_dir}")
    if plot_dir != Path("") and not os.access(plot_dir, os.W_OK):
        raise PermissionError(f"Plot directory is not writable: {plot_dir}")

    # --- Read data ---
    data = _read_input(input_file)
    F = int(data["F"])
    M = int(data["M"])
    H = int(data["H"])
    mu_0 = np.array(data["mu_0"], dtype=float)
    mu = np.array(data["mu"], dtype=float)
    x = np.array(data["x"], dtype=float)

    # --- Build grid values ---
    # Grid has F rows and (2H + 1) columns
    # Column 0 = mu_0, columns 1..2H = mu
    n_cols = 2 * H + 1
    grid = np.zeros((F, n_cols))
    grid[:, 0] = mu_0
    grid[:, 1:] = mu  # mu is F x 2H

    # --- Create plot ---
    cmap = mcolors.LinearSegmentedColormap.from_list("gr", ["green", "red"])
    fig, ax = plt.subplots(figsize=(max(n_cols * 0.8, 6), max(F * 0.8, 4)))

    ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="equal", origin="upper")

    # Draw grid lines and cell annotations
    for i in range(F):
        for k in range(n_cols):
            if k == 0:
                # First column: just show mu_0 value
                continue

            # k in the grid corresponds to time step k (1-based in grid, 0-based in x)
            k_x = k - 1  # index into x array (0-based, range 0..2H-1)

            if x[i, 0, k_x] == 1:
                # Maintenance: draw gear
                _draw_gear(ax, k, i)
            else:
                # Check if any j >= 1 is assigned
                assigned_j = None
                for j in range(1, M + 1):
                    if x[i, j, k_x] == 1:
                        assigned_j = j
                        break
                if assigned_j is not None:
                    ax.text(
                        k, i, str(assigned_j),
                        ha="center", va="center",
                        fontsize=10, fontweight="bold", color="black",
                    )
                else:
                    # Idle: draw sleeping cloud with "zzz"
                    _draw_sleep_cloud(ax, k, i)

    # Axis labels
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(range(0, n_cols))
    ax.set_yticks(range(F))
    ax.set_yticklabels(range(1, F + 1))
    ax.set_xlabel("Time step k")
    ax.set_ylabel("Flight i")

    # Grid lines between cells
    for i in range(F + 1):
        ax.axhline(i - 0.5, color="black", linewidth=0.5)
    for k in range(n_cols + 1):
        ax.axvline(k - 0.5, color="black", linewidth=0.5)

    plt.colorbar(ax.images[0], ax=ax, label="μ value", shrink=0.8)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def _resolve_plot_path(plot_file_path) -> Path:
    if plot_file_path is None:
        return Path("output.png")
    p = Path(plot_file_path)
    if p.suffix == "":
        p = p.with_suffix(".png")
    return p


def _read_input(input_file: Path) -> dict:
    ext = input_file.suffix.lower()
    if ext in (".yaml", ".yml"):
        with open(input_file, "r") as f:
            return yaml.safe_load(f)
    elif ext == ".json":
        with open(input_file, "r") as f:
            return json.load(f)
    elif ext in (".h5", ".hdf5"):
        return _read_hdf5(input_file)
    else:
        raise ValueError(f"Unsupported input file type: {ext}")


def _read_hdf5(path: Path) -> dict:
    data = {}
    scalar_keys = {"F", "H", "M"}
    array_keys = {"mu", "mu_0", "x"}

    with h5py.File(path, "r") as f:
        for key in scalar_keys:
            if key in f.attrs:
                data[key] = float(f.attrs[key])
            elif key in f:
                data[key] = float(f[key][()])
        for key in array_keys:
            if key in f:
                data[key] = f[key][()].tolist()

    return data


def _draw_gear(ax, cx, cy):
    """Draw a simple gear icon (Unicode) at cell (cx, cy)."""
    ax.text(
        cx, cy, "\u2699",
        ha="center", va="center",
        fontsize=16, color="black",
    )


def _draw_sleep_cloud(ax, cx, cy):
    """Draw a comic cloud with 'zzz' at cell (cx, cy)."""
    cloud = mpatches.FancyBboxPatch(
        (cx - 0.3, cy - 0.2), 0.6, 0.4,
        boxstyle="round,pad=0.05",
        facecolor="white", edgecolor="gray", linewidth=0.8, alpha=0.85,
    )
    ax.add_patch(cloud)
    ax.text(
        cx, cy, "zzz",
        ha="center", va="center",
        fontsize=8, fontstyle="italic", color="darkblue",
    )
