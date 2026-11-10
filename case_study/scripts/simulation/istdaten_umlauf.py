"""Rows, trips and UMLAUF_ID use per line for chosen operator lines in one Ist-Daten day.
Usage: python -I istdaten_umlauf.py <chunk.bin> <csize> OP:LINE [...]"""
import csv, io, struct, sys
from collections import defaultdict
data = open(sys.argv[1], "rb").read(); csize = int(sys.argv[2])
_, _, _, _, _, _, _, _, _, nlen, xlen = struct.unpack("<IHHHHHIIIHH", data[:30])
text = zlib = None
import zlib
text = zlib.decompressobj(-15).decompress(data[30 + nlen + xlen:30 + nlen + xlen + csize]).decode("utf-8-sig", "replace")
rows = csv.reader(io.StringIO(text), delimiter=";"); head = next(rows); ix = {h: i for i, h in enumerate(head)}
want = {tuple(s.split(":")) for s in sys.argv[3:]}
st = defaultdict(lambda: {"rows": 0, "trips": set(), "uml": set(), "uml_rows": 0, "real_an": 0, "an": 0})
for r in rows:
    if len(r) < len(head):
        continue
    k = (r[ix["BETREIBER_ABK"]].strip(), r[ix["LINIEN_TEXT"]])
    if k in want:
        s = st[k]; s["rows"] += 1; s["trips"].add(r[ix["FAHRT_BEZEICHNER"]])
        if r[ix["UMLAUF_ID"]]:
            s["uml_rows"] += 1; s["uml"].add(r[ix["UMLAUF_ID"]])
        if r[ix["ANKUNFTSZEIT"]]:
            s["an"] += 1
            if r[ix["AN_PROGNOSE_STATUS"]] == "REAL":
                s["real_an"] += 1
for k in sorted(st):
    s = st[k]
    print(k, "rows", s["rows"], "trips", len(s["trips"]), "rows with UMLAUF_ID", s["uml_rows"], "distinct UMLAUF_ID", len(s["uml"]),
          "arrivals", s["an"], "of which REAL", s["real_an"])
