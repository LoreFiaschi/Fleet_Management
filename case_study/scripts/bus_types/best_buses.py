"""Electric buses in the Swiss vehicle stock file BEST_ANDERE.txt (ASTRA IVZ open data).
Usage: python -E best_buses.py BEST_ANDERE.txt out_rows.tsv out_stats.txt
Selects rows whose vehicle kind is a bus (Gesellschaftswagen, Gelenkbus) and whose fuel is electric.
Writes the selected rows (key columns) and per type-approval statistics of the registered empty mass."""
import sys
import collections
import statistics

src, out_rows, out_stats = sys.argv[1], sys.argv[2], sys.argv[3]
with open(src, "r", encoding="utf-8", errors="replace", newline="") as f:
    head = f.readline().rstrip("\r\n").split("\t")
    idx = {h: i for i, h in enumerate(head)}
    need = ["Fahrzeugart", "Marke", "Marke_und_Typ", "Typengenehmigungs-Nr", "Sitzplätze", "Stehplätze",
            "Leergewicht", "Gesamtgewicht", "Leistung", "Treibstoff", "Achsen", "Erstinverkehrsetzung_Jahr",
            "Erstinverkehrsetzung_Kanton", "Inverkehrsetzung_Status_Code", "Inverkehrsetzung_Kanton", "PLZ",
            "Datenstand"]
    missing = [n for n in need if n not in idx]
    print("missing columns:", missing)
    kinds = collections.Counter()
    fuels = collections.Counter()
    rows = []
    n = 0
    for line in f:
        n += 1
        c = line.rstrip("\r\n").split("\t")
        if len(c) < len(head) - 2:
            continue
        art = c[idx["Fahrzeugart"]]
        if "Gesellschaftswagen" not in art and "Gelenkbus" not in art:
            continue
        kinds[art] += 1
        fu = c[idx["Treibstoff"]]
        fuels[fu] += 1
        if fu.strip() != "Elektrisch":
            continue
        rows.append([c[idx[k]] for k in need])
print("rows read", n, "bus kinds", kinds.most_common(), "fuels", fuels.most_common(12))
print("electric bus rows", len(rows))
with open(out_rows, "w", encoding="utf-8", newline="\n") as g:
    g.write("\t".join(need) + "\n")
    for r in sorted(rows, key=lambda r: (r[3], r[11])):
        g.write("\t".join(r) + "\n")
by_tg = collections.defaultdict(list)
for r in rows:
    by_tg[(r[1], r[2], r[3])].append(r)
with open(out_stats, "w", encoding="utf-8", newline="\n") as g:
    g.write("brand | make_type | TG | n | empty mass min/q1/median/q3/max kg | gross kg (values) | seats | standing | "
            "first reg. years | cantons (status I) | status codes\n")
    for key in sorted(by_tg, key=lambda k: (k[0], k[2])):
        rr = by_tg[key]
        lw = sorted(int(r[6]) for r in rr if r[6].strip().isdigit() and int(r[6]) > 0)
        if len(lw) >= 2:
            q = statistics.quantiles(lw, n=4)
            lws = "%d/%d/%d/%d/%d" % (lw[0], q[0], statistics.median(lw), q[2], lw[-1])
        elif lw:
            lws = str(lw[0])
        else:
            lws = "none"
        gw = collections.Counter(r[7] for r in rr)
        seats = collections.Counter(r[4] for r in rr)
        stand = collections.Counter(r[5] for r in rr)
        years = collections.Counter(r[11] for r in rr)
        cant = collections.Counter(r[14] for r in rr)
        stat = collections.Counter(r[13] for r in rr)
        g.write("%s | %s | %s | %d | %s | %s | %s | %s | %s | %s | %s\n" % (
            key[0], key[1], key[2], len(rr), lws, dict(gw.most_common(4)), dict(seats.most_common(3)),
            dict(stand.most_common(3)), dict(sorted(years.items())), dict(cant.most_common(8)),
            dict(stat.most_common())))
print("type approvals:", len(by_tg))
