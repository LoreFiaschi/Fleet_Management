"""Convert src/fleet_management/__init__.py to CRLF line endings (plan Task B, step B2)."""
import sys

path = sys.argv[1]
data = open(path, "rb").read()
data = data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
open(path, "wb").write(data)
print(data.count(b"\r\n"), data.count(b"\n"))
