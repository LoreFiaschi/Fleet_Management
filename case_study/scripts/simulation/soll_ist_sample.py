"""Count rows, lines, vehicles and delays in a sample of the VBZ Fahrzeiten SOLL-IST file.
Usage: python -I soll_ist_sample.py <csv sample>  (the last, cut row is dropped)"""
import csv, io, statistics, sys
from collections import Counter
raw = open(sys.argv[1], "rb").read()
raw = raw[:raw.rfind(b"\n") + 1]
rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
print("rows (stop-to-stop records):", len(rows), "columns:", len(rows[0]))
lines = Counter(r["linie"] for r in rows)
print("distinct lines:", len(lines), sorted(lines, key=lambda x: int(x) if x.isdigit() else 0))
print("distinct vehicles (fahrzeug):", len({r["fahrzeug"] for r in rows}), "distinct umlauf_von:", len({r["umlauf_von"] for r in rows}))
print("operating dates:", Counter(r["betriebsdatum"] for r in rows).most_common())
dev = [int(r["ist_an_nach1"]) - int(r["soll_an_nach"]) for r in rows]
q = statistics.quantiles(dev, n=20)
print("arrival delay at the next stop [s]: median %d, 5%% %d, 95%% %d" % (statistics.median(dev), q[0], q[-1]))
run = [(int(r["ist_an_nach1"]) - int(r["ist_ab_von"])) - (int(r["soll_an_nach"]) - int(r["soll_ab_von"])) for r in rows]
q = statistics.quantiles(run, n=20)
print("running time actual - planned between two stops [s]: median %d, 5%% %d, 95%% %d" % (statistics.median(run), q[0], q[-1]))
for ln in ("31", "33", "70", "89", "768", "759"):
    sub = [r for r in rows if r["linie"] == ln]
    if sub:
        print("line", ln, "records", len(sub), "vehicles", len({r["fahrzeug"] for r in sub}))
