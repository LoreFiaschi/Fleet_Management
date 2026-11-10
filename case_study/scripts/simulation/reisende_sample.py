"""Count the VBZ passenger-count sample (REISENDE.csv, first bytes) and the line table.
Usage: python -I reisende_sample.py <reisende sample> <LINIE.csv>"""
import csv, io, statistics, sys
from collections import Counter
raw = open(sys.argv[1], "rb").read(); raw = raw[:raw.rfind(b"\n") + 1]
rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig")), delimiter=";"))
print("rows:", len(rows), "columns:", len(rows[0]))
print("lines in sample:", sorted(Counter(r["Linienname"] for r in rows).items()))
print("day types in sample:", Counter(r["Tagtyp_Id"] for r in rows).most_common(5))
occ = [float(r["Besetzung"]) for r in rows if r["Besetzung"]]
print("occupancy (Besetzung) per segment: n %d, median %.1f, max %.1f" % (len(occ), statistics.median(occ), max(occ)))
mes = [int(r["Anzahl_Messungen"]) for r in rows]
print("measurements per trip and segment: median %d, min %d, max %d" % (statistics.median(mes), min(mes), max(mes)))
lin = list(csv.DictReader(open(sys.argv[2], encoding="utf-8-sig"), delimiter=";"))
print("LINIE.csv rows:", len(lin), "VSYS counts:", Counter(r["VSYS"] for r in lin).most_common())
names = {r["Linienname"] for r in lin}
for ln in ("31", "33", "39", "70", "89", "165", "759", "768", "787", "743"):
    print(ln, ln in names, end="; ")
print()
