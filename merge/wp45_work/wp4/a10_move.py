"""Step A10: move _derive before _check_models in config.py (definition order = call order).
Keeps CRLF endings. Run once."""
from pathlib import Path

CONFIG = Path(r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src\fleet_management\config.py")
lines = CONFIG.read_bytes().decode("utf-8").split("\r\n")


def idx(text):
    hits = [n for n, line in enumerate(lines) if line == text]
    assert len(hits) == 1, (text, hits)
    return hits[0]


start = idx("def _derive(cfg: FleetConfig) -> None:")
end = idx("def _quad_coefficients(cfg: FleetConfig, m: dict) -> tuple:")
target = idx("def _check_models(cfg: FleetConfig) -> None:")
assert target < start < end
block = lines[start:end]          # ends with two blank lines
new = lines[:target] + block + lines[target:start] + lines[end:]
assert len(new) == len(lines)
CONFIG.write_bytes("\r\n".join(new).encode("utf-8"))
print("moved", end - start, "lines")
