"""Decompress the first member of an Ist-Daten zip chunk and count rows by operator; keep VBG/VBZ samples.
Usage: python -I istdaten_scan.py <chunk.bin> <csize> <out_dir>"""
import csv, io, os, struct, sys, zlib
from collections import Counter
data = open(sys.argv[1], "rb").read(); csize = int(sys.argv[2]); out = sys.argv[3]
os.makedirs(out, exist_ok=True)
sig, _, _, method, _, _, _, _, _, nlen, xlen = struct.unpack("<IHHHHHIIIHH", data[:30])
name = data[30:30 + nlen].decode()
raw = data[30 + nlen + xlen:30 + nlen + xlen + csize]
text = zlib.decompressobj(-15).decompress(raw).decode("utf-8-sig", "replace")
rows = csv.reader(io.StringIO(text), delimiter=";")
head = next(rows)
print(name, "columns:", len(head)); print(head)
ix = {h: i for i, h in enumerate(head)}
op = Counter(); lines = Counter(); n = 0; keep = []
for r in rows:
    if len(r) < len(head):
        continue
    n += 1
    a = r[ix["BETREIBER_ABK"]]
    op[a] += 1
    if a in ("VBG", "VBZ") and r[ix["PRODUKT_ID"]] == "Bus":
        lines[(a, r[ix["LINIEN_TEXT"]])] += 1
        if r[ix["LINIEN_TEXT"]] in ("759", "768", "31", "89") and len(keep) < 4000:
            keep.append(r)
print("rows:", n, "operators:", len(op))
print("top operators:", op.most_common(8))
print("VBG rows:", op.get("VBG"), "VBZ rows:", op.get("VBZ"))
print("VBG bus lines:", sorted((l, c) for (a, l), c in lines.items() if a == "VBG")[:40])
w = csv.writer(open(os.path.join(out, "istdaten_20261001_sample_759_768_31_89.csv"), "w", newline="", encoding="utf-8"))
w.writerow(head); w.writerows(keep)
for ln in ("759", "768"):
    sub = [r for r in keep if r[ix["LINIEN_TEXT"]] == ln]
    st = Counter(r[ix["AN_PROGNOSE_STATUS"]] for r in sub)
    print("line", ln, "rows", len(sub), "trips", len({r[ix["FAHRT_BEZEICHNER"]] for r in sub}), "AN_PROGNOSE_STATUS", st.most_common())
if keep:
    print("example row:", dict(zip(head, keep[0])))
