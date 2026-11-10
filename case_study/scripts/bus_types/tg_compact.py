"""Compact listing of the electric bus type approvals (from tg_ebus_full.txt written by tg_buses.py).
Usage: python -E tg_compact.py tg_ebus_full.txt out.txt
Keeps the fields that matter for the bus-type table."""
import sys
import re

WANT = {0, 1, 6, 10, 11, 12, 22, 25, 59, 60, 64, 66, 81, 82, 85, 92, 93, 102, 103, 104, 105,
        144, 145, 146, 147, 148, 149, 150, 151, 156, 157, 158, 159, 160, 161, 162, 163, 190, 191, 192}
recs = []
cur = None
for line in open(sys.argv[1], encoding="utf-8"):
    if line.startswith("===="):
        cur = []
        recs.append(cur)
        continue
    m = re.match(r"\s+\[(\d+)\] ([^:]*): (.*)", line.rstrip("\n"))
    if m and cur is not None:
        i = int(m.group(1))
        if i in WANT or (i >= 199 and re.search(r"Leistung|kWh|Batter|Akku|Gewicht|Achs", m.group(3))):
            cur.append("[%d] %s" % (i, m.group(3)))
with open(sys.argv[2], "w", encoding="utf-8", newline="\n") as g:
    for r in recs:
        g.write(" | ".join(r) + "\n\n")
print(len(recs), "records")
