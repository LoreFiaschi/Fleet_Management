"""Count what the ZVV GTFS (data.stadt-zuerich.ch) holds for chosen lines: routes, trips, shapes.
Usage: python -I zvv_gtfs_check.py <dir with agency.txt routes.txt trips.txt shapes.txt> <line,line,...>"""
import csv
import math
import sys
from collections import Counter, defaultdict

d = sys.argv[1]
lines = sys.argv[2].split(",")


def rows(name):
    with open(d + "/" + name, encoding="utf-8-sig", newline="") as f:
        yield from csv.DictReader(f)


print("agency:", [(r["agency_id"], r["agency_name"]) for r in rows("agency.txt")])
routes = list(rows("routes.txt"))
print("routes:", len(routes), "columns:", list(routes[0].keys()))
print("route_type counts:", Counter(r["route_type"] for r in routes).most_common())
print("agency_id counts in routes:", Counter(r.get("agency_id") for r in routes).most_common(8))
sel = {r["route_id"]: r for r in routes if r["route_short_name"] in lines}
for rid, r in sel.items():
    print("  route", rid, r["route_short_name"], r.get("route_long_name"), "agency", r.get("agency_id"), "type", r["route_type"])
missing = set(lines) - {r["route_short_name"] for r in sel.values()}
print("lines not found:", sorted(missing))

trips_cols = None
shape_by_line = defaultdict(Counter)
n_trips = Counter()
no_shape = 0
all_trips = 0
for t in rows("trips.txt"):
    all_trips += 1
    if trips_cols is None:
        trips_cols = list(t.keys())
    if not t.get("shape_id"):
        no_shape += 1
    if t["route_id"] in sel:
        ln = sel[t["route_id"]]["route_short_name"]
        n_trips[ln] += 1
        shape_by_line[ln][t.get("shape_id", "")] += 1
print("trips columns:", trips_cols)
print("trips:", all_trips, "without shape_id:", no_shape)

need = {s for c in shape_by_line.values() for s in c}
pts = defaultdict(list)
shape_cols = None
n_shapes = set()
for s in rows("shapes.txt"):
    if shape_cols is None:
        shape_cols = list(s.keys())
    n_shapes.add(s["shape_id"])
    if s["shape_id"] in need:
        pts[s["shape_id"]].append((int(s["shape_pt_sequence"]), float(s["shape_pt_lat"]), float(s["shape_pt_lon"])))
print("shapes columns:", shape_cols, "distinct shapes:", len(n_shapes))


def length_km(p):
    p = sorted(p)
    L = 0.0
    for (_, a, b), (_, c, e) in zip(p, p[1:]):
        x = math.radians(e - b) * math.cos(math.radians((a + c) / 2))
        y = math.radians(c - a)
        L += 6371.0 * math.hypot(x, y)
    return L, len(p)


for ln in lines:
    if ln not in shape_by_line:
        continue
    top = shape_by_line[ln].most_common(2)
    desc = []
    for sid, cnt in top:
        if sid and sid in pts:
            L, n = length_km(pts[sid])
            desc.append("%s: %d trips, %d points, %.2f km" % (sid, cnt, n, L))
        else:
            desc.append("%r: %d trips, no points" % (sid, cnt))
    print("line", ln, "trips", n_trips[ln], "distinct shapes", len(shape_by_line[ln]), "|", " ; ".join(desc))
