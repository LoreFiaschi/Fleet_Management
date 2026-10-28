"""Replace lines [start, end] (1-based, inclusive) of base.py by the text of a file.

Usage: python splice.py START END NEWFILE
The file base.py keeps its CRLF line endings. Before writing, the script checks
that the first and last replaced lines start with the strings given in
--expect-first / --expect-last (optional safety check).
"""
import argparse
import pathlib

BASE = pathlib.Path(
    r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge"
    r"\src\fleet_management\degradation_model\base.py"
)

parser = argparse.ArgumentParser()
parser.add_argument("start", type=int)
parser.add_argument("end", type=int)
parser.add_argument("newfile")
parser.add_argument("--expect-first", default=None)
parser.add_argument("--expect-last", default=None)
args = parser.parse_args()

raw = BASE.read_bytes().decode("utf-8")
lines = raw.split("\r\n")
if lines[-1] == "":
    lines.pop()
first, last = lines[args.start - 1], lines[args.end - 1]
if args.expect_first is not None and not first.startswith(args.expect_first):
    raise SystemExit(f"first line mismatch: {first!r}")
if args.expect_last is not None and not last.startswith(args.expect_last):
    raise SystemExit(f"last line mismatch: {last!r}")
new_text = pathlib.Path(args.newfile).read_bytes().decode("utf-8").replace("\r\n", "\n")
new_lines = new_text.split("\n")
if new_lines[-1] == "":
    new_lines.pop()
lines[args.start - 1:args.end] = new_lines
BASE.write_bytes(("\r\n".join(lines) + "\r\n").encode("utf-8"))
print(f"replaced lines {args.start}-{args.end} ({args.end - args.start + 1} lines) by {len(new_lines)} lines")
