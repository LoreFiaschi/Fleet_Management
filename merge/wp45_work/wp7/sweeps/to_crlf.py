"""Convert the files named on the command line to CRLF line endings, by bytes (no round trip through text mode)."""
import sys

for name in sys.argv[1:]:
    with open(name, "rb") as f:
        raw = f.read()
    text = raw.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    with open(name, "wb") as f:
        f.write(text)
    print(name, raw.count(b"\n"), "lines;", "non-ASCII bytes:", sum(1 for b in text if b > 127))
