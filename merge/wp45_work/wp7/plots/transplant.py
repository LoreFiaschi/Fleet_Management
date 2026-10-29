"""Transplant top-level blocks of the prototype into plotter.py (Task A tool; text is copied verbatim).

Usage: python transplant.py <op> [<op> ...]
  header                 replace the header (everything before the first def) by the prototype's
  replace:OLD=NEW        replace the block OLD of plotter.py by the block NEW of the prototype
  delete:NAME            delete the block NAME of plotter.py
  insert:NEW@AFTER       insert the prototype block NEW after the block AFTER of plotter.py
The file keeps CRLF line endings. Blocks are separated by two blank lines.
"""
import re
import sys
from pathlib import Path

TARGET = Path(r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\src\fleet_management\utils\plotter.py")
PROTO = Path(r"C:\Users\utente\AppData\Local\Temp\claude\C--Users-utente-Documents-GitHub-Fleet-Management\ccdc4317-924c-4e98-951b-d5a88989e5f4\scratchpad\wp7\plots\proto_plotter.py")


def load(path):
    text = path.read_bytes().decode("utf-8").replace("\r\n", "\n")
    parts = text.rstrip("\n").split("\n\n\ndef ")
    header = parts[0]
    blocks = []
    for part in parts[1:]:
        body = "def " + part
        blocks.append((re.match(r"def (\w+)", body).group(1), body))
    return header, blocks


def save(path, header, blocks):
    text = header + "".join("\n\n\n" + body for _, body in blocks) + "\n"
    path.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))


def index(blocks, name):
    return next(i for i, (n, _) in enumerate(blocks) if n == name)


def main(ops):
    header, blocks = load(TARGET)
    p_header, p_blocks = load(PROTO)
    p_by_name = dict(p_blocks)
    for op in ops:
        kind, _, arg = op.partition(":")
        if op == "header":
            header = p_header
        elif kind == "replace":
            old, new = arg.split("=")
            blocks[index(blocks, old)] = (new, p_by_name[new])
        elif kind == "delete":
            del blocks[index(blocks, arg)]
        elif kind == "insert":
            new, after = arg.split("@")
            blocks.insert(index(blocks, after) + 1, (new, p_by_name[new]))
        else:
            raise SystemExit(f"unknown op {op}")
    save(TARGET, header, blocks)
    print("blocks:", [n for n, _ in blocks])


if __name__ == "__main__":
    main(sys.argv[1:])
