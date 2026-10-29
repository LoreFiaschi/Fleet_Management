"""Build tests/test_plotter.py from line ranges of the prototype (LF endings).

Usage: python build_tests.py <first line> <last line> [append]
Without "append" the target is created; with it the range is appended.
"""
import sys
from pathlib import Path

PROTO = Path(r"C:\Users\utente\AppData\Local\Temp\claude\C--Users-utente-Documents-GitHub-Fleet-Management\ccdc4317-924c-4e98-951b-d5a88989e5f4\scratchpad\wp7\plots\proto_test_plotter.py")
TARGET = Path(r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge\tests\test_plotter.py")

first, last = int(sys.argv[1]), int(sys.argv[2])
lines = PROTO.read_bytes().decode("utf-8").replace("\r\n", "\n").split("\n")
chunk = "\n".join(lines[first - 1:last]) + "\n"
if len(sys.argv) > 3 and sys.argv[3] == "append":
    old = TARGET.read_bytes().decode("utf-8")
    TARGET.write_bytes((old + chunk).encode("utf-8"))
else:
    TARGET.write_bytes(chunk.encode("utf-8"))
print("lines", first, "to", last, "->", TARGET)
