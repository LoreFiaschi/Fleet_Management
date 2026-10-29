import re, sys
p = "proto_plotter.py"
s = open(p, encoding="utf-8", newline="").read()
nl = "\r\n"
def rep(old, new):
    global s
    old = old.replace("\n", nl); new = new.replace("\n", nl)
    assert s.count(old) == 1, old[:60]
    s = s.replace(old, new)
rep('def _draw_horizon_sweep(view: dict[str, Any], plot_path: Path) -> None:',
    'def _draw_horizon_sweep(view: dict[str, Any]):\n    """Return the figure of the sweep plot: panels (a), (b), (c) of the spec."""')
rep('label="proven optimal", zorder=3)', 'label="proven (optimal)", zorder=3)')
rep('label="feasible", zorder=3)', 'label="feasible, not proven", zorder=3)')
rep('label="no feasible solution found", zorder=4,', 'label="no feasible solution", zorder=4,')
start = s.index('    gap_tolerance_percent = (100.0 * view["maximum_mip_gap_for_stopping"])')
end = s.index('    if np.any(finite_gaps):\r\n        gap_percent = 100.0 * gaps\r\n        gap_ax.plot(h2, gap_percent')
new = '''    # Reference of panel (b): the stopping gap if it is a number, otherwise the
    # requested MIP gap of each case.
    if view["maximum_mip_gap_for_stopping"] is not None:
        stopping_percent = 100.0 * view["maximum_mip_gap_for_stopping"]
        gap_ax.axhline(
            stopping_percent, color="#15803d", linestyle="--", linewidth=1.3,
            label=f"{stopping_percent:g}% stopping gap", zorder=2,
        )
    else:
        requested = [
            (row["H2"], 100.0 * row["requested_mip_gap"]) for row in rows
            if row["requested_mip_gap"] is not None
        ]
        if requested:
            gap_ax.scatter(
                [h for h, _ in requested], [g for _, g in requested],
                s=160, marker="_", color="#15803d", linewidth=1.6,
                label="requested MIP gap", zorder=2,
            )
    finite_gaps = feasible & np.isfinite(gaps)
'''.replace("\n", nl)
s = s[:start] + new + s[end:]
rep('label="proven optimal", zorder=3,\n        )', 'label="proven (optimal)", zorder=3,\n        )')
rep('label="time-limited feasible", zorder=3,', 'label="feasible, not proven", zorder=3,')
rep('label="no feasible solution; gap undefined", zorder=4,', 'label="no feasible solution", zorder=4,')
rep('''    figure.suptitle("Operating-horizon sweep", x=0.02, y=0.975, ha="left",
                    fontsize=16, fontweight="bold")
    figure.text(0.02, 0.925, dimensions, fontsize=9, color="#4b5563")
''', '''    figure.suptitle(f"Operating-horizon sweep ({dimensions})", x=0.02, y=0.975,
                    ha="left", fontsize=16, fontweight="bold")
''')
rep('''        wspace=0.32, hspace=0.42
    )
    figure.savefig(plot_path, dpi=170, bbox_inches="tight")
    plt.close(figure)
''', '''        wspace=0.32, hspace=0.42
    )
    return figure
''')
open(p, "w", encoding="utf-8", newline="").write(s)
print("ok")
