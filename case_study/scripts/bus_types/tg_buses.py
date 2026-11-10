"""Filter the ASTRA type-approval file (TG-Automobil.txt) for electric buses.
Usage: python -E tg_buses.py TG-Automobil.txt out_summary.tsv out_full.txt
- out_summary.tsv: one row per type approval with the key columns
- out_full.txt: every non-empty field of each selected row (for reading)
Counts are printed: all rows, bus rows, electric bus rows, by brand."""
import sys
import collections

src, out_sum, out_full = sys.argv[1], sys.argv[2], sys.argv[3]
KEY = {0: "TG", 1: "erteilt", 6: "Fahrzeugart", 9: "Klasse", 10: "Marke", 11: "Typ", 12: "Variante",
       59: "MotorMarke", 60: "MotorTyp", 61: "Bauart", 62: "Treibstoff", 64: "kW", 81: "PlaetzeVon",
       82: "PlaetzeBis", 85: "Tueren", 93: "LaengeVon", 94: "LaengeBis", 144: "LeerVon", 145: "LeerBis",
       146: "GarantieVon", 147: "GarantieBis", 148: "AchsVvon", 149: "AchsVbis", 150: "AchsHvon",
       151: "AchsHbis", 190: "Reifen69", 191: "Reifen70", 192: "Reifen71"}
n_all = 0
n_bus = 0
fuels = collections.Counter()
brands = collections.Counter()
rows = []
with open(src, "r", encoding="cp1252", errors="replace", newline="") as f:
    head = f.readline().rstrip("\r\n").split("\t")
    for line in f:
        n_all += 1
        c = line.rstrip("\r\n").split("\t")
        if len(c) < 200:
            continue
        art = c[6].upper()
        if "GESELLSCHAFTSWAGEN" not in art and "GELENKBUS" not in art:
            continue
        n_bus += 1
        fuels[c[62]] += 1
        if "LEKTR" not in (c[62] + c[61]).upper():
            continue
        brands[c[10]] += 1
        rows.append(c)
print("rows", n_all, "bus rows", n_bus, "electric bus rows", len(rows))
print("fuel values of bus rows:", fuels.most_common(15))
print("brands of electric bus rows:", brands.most_common(60))
with open(out_sum, "w", encoding="utf-8", newline="\n") as g:
    g.write("\t".join(KEY.values()) + "\n")
    for c in sorted(rows, key=lambda r: (r[10], r[1])):
        g.write("\t".join(c[i].strip() if i < len(c) else "" for i in KEY) + "\n")
with open(out_full, "w", encoding="utf-8", newline="\n") as g:
    for c in sorted(rows, key=lambda r: (r[10], r[1])):
        g.write("==== %s %s %s %s\n" % (c[0], c[1], c[10], c[11]))
        for i, v in enumerate(c):
            v = v.strip()
            if v and v != "0":
                name = head[i] if i < len(head) and head[i] else "col%d" % i
                g.write("  [%d] %s: %s\n" % (i, name, v))
