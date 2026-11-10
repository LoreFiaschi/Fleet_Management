"""Share of trips with block_id, and trips per block, in a GTFS trips.txt (by route prefix).
Usage: python -I block_check.py <trips.txt>"""
import csv, sys
from collections import Counter, defaultdict
filled = Counter(); total = Counter(); per_block = defaultdict(Counter)
with open(sys.argv[1], encoding="utf-8-sig", newline="") as f:
    for t in csv.DictReader(f):
        p = t["route_id"].split("-")[0]
        total[p] += 1
        if t["block_id"]:
            filled[p] += 1
            per_block[p][t["block_id"]] += 1
for p, n in total.most_common(12):
    b = per_block[p]
    sizes = Counter(b.values())
    print("prefix", p, "trips", n, "with block_id", filled[p], "blocks", len(b), "trips per block (size:count)", sorted(sizes.items())[:8])
