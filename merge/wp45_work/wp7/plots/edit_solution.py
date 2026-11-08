"""Apply the planned schedule-plot edits to the scratchpad prototype (not the worktree)."""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
P = HERE / "proto_plotter.py"
s = P.read_text(encoding="utf-8").replace("\r\n", "\n")


def functions(text):
    """Map of top-level function name -> (start, end) offsets, end at the next top-level def."""
    starts = [(m.group(1), m.start()) for m in re.finditer(r"^def (\w+)", text, flags=re.M)]
    out = {}
    for n, (name, start) in enumerate(starts):
        end = starts[n + 1][1] if n + 1 < len(starts) else len(text)
        out[name] = (start, end)
    return out


def replace_functions(names, block):
    """Replace the consecutive functions ``names`` (first to last) by ``block``."""
    global s
    f = functions(s)
    start, end = f[names[0]][0], f[names[-1]][1]
    s = s[:start] + block.rstrip("\n") + "\n\n\n" + s[end:]


def remove_function(name):
    global s
    start, end = functions(s)[name]
    s = s[:start] + s[end:]


def rep(old, new):
    global s
    assert s.count(old) == 1, old[:70]
    s = s.replace(old, new)


replace_functions(["_normalise_solution"], (HERE / "block_normalise.py").read_text(encoding="utf-8"))
replace_functions(
    ["_component_action_label", "_draw_statistics", "_model_grid", "_component_names",
     "_component_row_label", "_model_abbreviation", "_unique_models"],
    (HERE / "block_stats.py").read_text(encoding="utf-8"),
)
remove_function("_cell_grid")
remove_function("_legacy_h1")
replace_functions(["_read_input", "_read_solution_input"], (HERE / "block_log.py").read_text(encoding="utf-8"))
remove_function("_read_hdf5")

# drawing of the schedule plot
rep('def _draw_solution(view: dict[str, Any], plot_path: Path) -> None:\n',
    'def _draw_solution(view: dict[str, Any]):\n'
    '    """Return the figure of the schedule plot: the grid and the summary panel."""\n')
rep('ax.set_xticklabels(["initial", *range(T)],', 'ax.set_xticklabels(["initial", *range(1, T + 1)],')
rep('ax.set_xlabel("Time step")', 'ax.set_xlabel("Day")')
rep('ax.set_ylabel("Vehicle and component model")', 'ax.set_ylabel("Train and component")')
rep('ax.set_title("Schedule and physical expected damage", loc="left", pad=42,',
    'ax.set_title("Schedule and tracked mean degradation", loc="left", pad=42,')
rep('(H1 + 1) / 2, 1.015, "initialization",', '(H1 + 1) / 2, 1.015, "transitory",')
rep('label="physical mean / failure threshold",', 'label="tracked mean / failure threshold",')
rep('''               for label, colour in (("M$_j$ mission", ACTION_COLOURS["mission"]),
                                     ("I idle", ACTION_COLOURS["idle"]),
                                     ("R repair", ACTION_COLOURS["repair"]),
                                     ("P replacement", ACTION_COLOURS["replacement"]))]''',
    '''               for label, colour in (("P replacement", ACTION_COLOURS["replacement"]),
                                     ("R repair", ACTION_COLOURS["repair"]),
                                     ("M$_j$ mission $j$", ACTION_COLOURS["mission"]),
                                     ("D depot, no intervention", ACTION_COLOURS["depot"]),
                                     ("I idle train", ACTION_COLOURS["idle"]))]''')
rep('''                 fontsize=15, fontweight="bold")
    fig.savefig(plot_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
''', '''                 fontsize=15, fontweight="bold")
    return fig
''')
P.write_text(s.replace("\n", "\r\n"), encoding="utf-8", newline="")
print("ok")
