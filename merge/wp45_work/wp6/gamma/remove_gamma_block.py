"""Remove the old Gamma block from base.py (WP6 Gamma plan, step 2).

Usage: python remove_gamma_block.py <path of base.py>
Removes (1) the paragraph of the module docstring that announces the Gamma block (3 lines and
the blank line after them) and (2) the block itself: from the banner line before
"GAMMA FINITE-HORIZON TAIL BLOCK" to the two blank lines after
register_cell_builder("gamma", GammaCellBuilder()). Every marker is checked first; on any
mismatch the file is not written. Line endings are kept.
"""
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = path.read_bytes()
newline = b"\r\n" if b"\r\n" in data else b"\n"
lines = data.split(newline)


def text(i):
    return lines[i].decode("utf-8")


def find(marker, exact=False):
    hits = [i for i in range(len(lines)) if (text(i).strip() == marker if exact else marker in text(i))]
    if len(hits) != 1:
        sys.exit(f"marker {marker!r} found {len(hits)} times; nothing written")
    return hits[0]


doc = find("The Gamma block below stays until the Gamma cell builder is rewritten (work", exact=True)
if not (text(doc + 1).startswith("package WP6 of the merge)") and text(doc + 2).strip() == "Gamma cell fails when it is built."
        and text(doc + 3).strip() == ""):
    sys.exit("docstring paragraph differs from the expected text; nothing written")

title = find("GAMMA FINITE-HORIZON TAIL BLOCK")
start = title - 1
registration = find('register_cell_builder("gamma", GammaCellBuilder())', exact=True)
end = registration + 2
if not (text(start).startswith("# ====") and text(title + 1).startswith("# ====")):
    sys.exit("banner of the Gamma block differs; nothing written")
if not (text(registration + 1) == "" and text(registration + 2) == ""
        and text(registration + 3).startswith("# ====")
        and text(registration + 4).strip() == "# Run options and solver settings"):
    sys.exit("lines after the registration differ; nothing written")
if not doc < start < registration:
    sys.exit("markers out of order; nothing written")

kept = lines[:doc] + lines[doc + 4:start] + lines[end + 1:]
result = newline.join(kept)
for name in ("GammaCellBuilder", "_add_binary_scaled_product", "_gamma_reachable_upper_bounds",
             "GAMMA FINITE-HORIZON"):
    if name.encode() in result:
        sys.exit(f"{name} still present after the cut; nothing written")
path.write_bytes(result)
print(f"removed docstring lines {doc + 1}-{doc + 4} and block lines {start + 1}-{end + 1} "
      f"({end - start + 1} lines); {len(lines) - 1} -> {len(kept) - 1} lines")
