"""Step A5: replace FleetConfig and load_config, delete the old helpers and _validate_cells,
and reorder the sections of config.py.  Keeps CRLF line endings.  Run once."""
from pathlib import Path

SCRATCH = Path(__file__).resolve().parent
CONFIG = Path(r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src\fleet_management\config.py")

raw = CONFIG.read_bytes().decode("utf-8")
assert "\r\n" in raw and raw.count("\r\n") == raw.count("\n")
lines = raw.split("\r\n")
BANNER = "# " + "-" * 75


def idx(text):
    hits = [n for n, line in enumerate(lines) if line == text]
    assert len(hits) == 1, (text, hits)
    return hits[0]


def banner(title):
    n = idx("# " + title)
    assert lines[n - 1] == BANNER and lines[n + 1] == BANNER, title
    return n - 1


def strip(block):
    block = list(block)
    while block and block[-1] == "":
        block.pop()
    return block


def text_lines(name):
    return (SCRATCH / name).read_text(encoding="utf-8").rstrip("\n").split("\n")


head = lines[:banner("Normalized representation")]
scalars = lines[banner("Scalars"):banner("Horizon")]
parse_horizon = lines[idx("def _parse_horizon(H_raw):"):banner("Cells, selectors and profiles")]
infer_L = lines[idx("def _infer_L(data: dict, F: int) -> int:"):idx("def _run_options(data: dict, H1: int) -> dict:")]
component_names = lines[idx("def _component_names(value, L: int) -> list[str]:"):idx("def _component_cost(value, L: int, name: str) -> np.ndarray:")]
run_options = lines[idx("def _run_options(data: dict, H1: int) -> dict:"):banner("Cell-wise cross-field validation")]
cells = lines[banner("Cells, selectors and profiles"):banner("Broadcasting helpers  (new (F, L, ...) layout)")]
reader = lines[banner("Input reader"):]

parts = [
    strip(head),
    text_lines("a5_fleetconfig.txt"),
    strip(scalars),
    [BANNER, "# Sizes and names", BANNER] + [""][:0] + strip(parse_horizon) + ["", ""] + strip(infer_L)
    + ["", ""] + strip(component_names),
    [BANNER, "# Run options", BANNER] + strip(run_options),
    strip(cells),
    text_lines("a5_loader.txt"),
    strip(reader),
]
out = []
for n, part in enumerate(parts):
    if n:
        out += ["", ""]
    out += part
out.append("")
CONFIG.write_bytes("\r\n".join(out).encode("utf-8"))
print("lines:", len(lines), "->", len(out))
