"""Line features of one bus depot, from the Swiss GTFS timetable and the swisstopo height profiles.

The script does not download anything. It reads files that were downloaded before and are
given as command-line arguments. It has three steps:

1. filter   Read the Swiss GTFS zip and keep the trips of the chosen operator (agency).
            Writes small CSV files into a work folder.
               python -I extract_line_features.py filter <gtfs.zip> <work dir> --agency 773
2. geoms    For each line, find the main stop pattern per direction on a reference weekday.
            Optionally match it to an OpenStreetMap (OSM) bus route relation, read from an
            Overpass API answer (JSON, "out geom"), to get the road geometry. Write
            patterns.csv into the work folder, with one URL per pattern for the swisstopo
            profile service (geometry in Swiss LV95 coordinates, EPSG:2056).
               python -I extract_line_features.py geoms <work dir> --lines 759,768
                   --date 20260616 --osm <overpass.json>
            The user then downloads each URL into its own folder (not done by this script).
3. table    Build the table of line features (one row per line) from the work folder and the
            downloaded profiles, given in the order of the rows of patterns.csv.
               python -I extract_line_features.py table <work dir> <profile.json> [...] --out <csv>
                   --lines 759,768 --weekday 20260616 --saturday 20260620 --sunday 20260621

The OSM query used for the Bassersdorf lines (Overpass QL):
   [out:json][timeout:120];area["ISO3166-2"="CH-ZH"]->.a;
   rel(area.a)["route"="bus"]["ref"~"^(739|742|...|797)$"];out geom;

Definitions used in the table (columns _d0 and _d1 are the two directions):
- Main pattern: the most frequent sequence of stop places of a direction on the weekday.
- Route length: the OSM road between the first and the last stop, when an OSM relation fits
  the pattern (all stops in order, intermediate stops within 50 m of the road, termini within
  150 m, no gap between ways over 10 m); else the sum of the straight lines between consecutive
  stops, a lower bound. The GTFS feed has no road geometry (no shapes.txt).
- Running time: median over the weekday trips of the main pattern, first departure to last
  arrival. Mean speed: (length_d0 + length_d1) / (run_time_d0 + run_time_d1).
- Elevation: swisstopo profile (model COMB) along the same geometry, smoothed (see
  profile_stats).
- n_j: the minimum number of vehicles that runs the line's own timetable on that day, with a
  minimum layover between two trips (default 3 min) and empty runs between stop places at
  20 km/h over 1.3 times the straight distance (min_fleet). The line is taken alone: no vehicle
  serves another line on the same day. n_cycle_wd is a cross-check from cycle time and headway.
- Vehicle-km: the sum over the day's trips of their lengths (stop-to-stop length times the road
  factor of the direction). Empty runs to and from the depot are not included.
- km per vehicle: vehicle-km / n_j.
The GTFS field block_id is not used: in this feed it mostly joins two trips that run through,
not the day-long vehicle blocks (German "Umlauf").
"""

import argparse
import csv
import io
import json
import math
import os
import sys
import zipfile
from collections import Counter, defaultdict
from urllib.parse import quote


# ---------------------------------------------------------------- small helpers

def read_member(z, name):
    """Yield the rows of one GTFS file inside the zip as dicts (UTF-8, BOM removed)."""
    with z.open(name) as f:
        for row in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8-sig", newline="")):
            yield row


def write_csv(path, rows, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def seconds(hms):
    """GTFS time 'HH:MM:SS' (HH may be 24 or more) to seconds after midnight."""
    h, m, s = hms.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres (sphere of radius 6,371,000 m)."""
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def wgs84_to_lv95(lat, lon):
    """WGS84 to Swiss LV95 (E, N), with the approximate formulas published by swisstopo
    ("Approximate formulas for the transformation between Swiss projection coordinates and
    WGS84"); accuracy about 1 m, enough for a height profile."""
    phi = (lat * 3600 - 169028.66) / 10000
    lam = (lon * 3600 - 26782.5) / 10000
    e = (2600072.37 + 211455.93 * lam - 10938.51 * lam * phi - 0.36 * lam * phi ** 2
         - 44.54 * lam ** 3)
    n = (1200147.07 + 308807.95 * phi + 3745.25 * lam ** 2 + 76.63 * phi ** 2
         - 194.56 * lam ** 2 * phi + 119.79 * phi ** 3)
    return e, n


# ---------------------------------------------------------------- step 1: filter

def step_filter(args):
    z = zipfile.ZipFile(args.gtfs)
    os.makedirs(args.work, exist_ok=True)

    routes = [r for r in read_member(z, "routes.txt") if r["agency_id"] == args.agency]
    route_ids = {r["route_id"] for r in routes}
    write_csv(os.path.join(args.work, "routes.csv"), routes,
              ["route_id", "agency_id", "route_short_name", "route_long_name", "route_type"])
    print("routes:", len(routes))

    trips = [t for t in read_member(z, "trips.txt") if t["route_id"] in route_ids]
    trip_ids = {t["trip_id"] for t in trips}
    service_ids = {t["service_id"] for t in trips}
    write_csv(os.path.join(args.work, "trips.csv"), trips,
              ["route_id", "service_id", "trip_id", "trip_headsign", "direction_id", "block_id"])
    print("trips:", len(trips))

    # stop_times.txt is large (about 1.8 GB unzipped): stream it and keep only our trips.
    stop_ids = set()
    with open(os.path.join(args.work, "stop_times.csv"), "w", encoding="utf-8", newline="") as f:
        fields = ["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        n = 0
        for st in read_member(z, "stop_times.txt"):
            if st["trip_id"] in trip_ids:
                w.writerow(st)
                stop_ids.add(st["stop_id"])
                n += 1
    print("stop_times:", n)

    stops = [s for s in read_member(z, "stops.txt") if s["stop_id"] in stop_ids]
    write_csv(os.path.join(args.work, "stops.csv"), stops,
              ["stop_id", "stop_name", "stop_lat", "stop_lon", "parent_station"])
    cal = [c for c in read_member(z, "calendar.txt") if c["service_id"] in service_ids]
    write_csv(os.path.join(args.work, "calendar.csv"), cal,
              ["service_id", "monday", "tuesday", "wednesday", "thursday", "friday",
               "saturday", "sunday", "start_date", "end_date"])
    cd = [c for c in read_member(z, "calendar_dates.txt") if c["service_id"] in service_ids]
    write_csv(os.path.join(args.work, "calendar_dates.csv"), cd,
              ["service_id", "date", "exception_type"])
    info = list(read_member(z, "feed_info.txt"))
    write_csv(os.path.join(args.work, "feed_info.csv"), info, list(info[0].keys()))
    print("stops:", len(stops), "calendar:", len(cal), "calendar_dates:", len(cd))


# ---------------------------------------------------------------- shared loading

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


class Work:
    """The filtered GTFS tables of the work folder."""

    def __init__(self, work):
        self.routes = {r["route_id"]: r for r in read_csv(os.path.join(work, "routes.csv"))}
        self.trips = read_csv(os.path.join(work, "trips.csv"))
        self.stops = {s["stop_id"]: s for s in read_csv(os.path.join(work, "stops.csv"))}
        self.cal = {c["service_id"]: c for c in read_csv(os.path.join(work, "calendar.csv"))}
        self.exc = defaultdict(dict)
        for c in read_csv(os.path.join(work, "calendar_dates.csv")):
            self.exc[c["service_id"]][c["date"]] = c["exception_type"]
        self.feed = read_csv(os.path.join(work, "feed_info.csv"))[0]
        self.st = defaultdict(list)
        for s in read_csv(os.path.join(work, "stop_times.csv")):
            self.st[s["trip_id"]].append(s)
        for t in self.st:
            self.st[t].sort(key=lambda s: int(s["stop_sequence"]))

    def runs_on(self, service_id, date):
        """GTFS rule: calendar_dates exceptions (1 = added, 2 = removed) override calendar."""
        e = self.exc.get(service_id, {}).get(date)
        if e == "1":
            return True
        if e == "2":
            return False
        c = self.cal.get(service_id)
        if c is None or not (c["start_date"] <= date <= c["end_date"]):
            return False
        import datetime
        wd = datetime.date(int(date[:4]), int(date[4:6]), int(date[6:])).weekday()
        return c[WEEKDAYS[wd]] == "1"

    def line_trips(self, line, date):
        """Trips of the line (route_short_name) that run on the date."""
        return [t for t in self.trips
                if self.routes[t["route_id"]]["route_short_name"] == line
                and self.runs_on(t["service_id"], date) and t["trip_id"] in self.st]

    def trip_length_m(self, trip_id):
        st = self.st[trip_id]
        d = 0.0
        for a, b in zip(st, st[1:]):
            sa, sb = self.stops[a["stop_id"]], self.stops[b["stop_id"]]
            d += haversine_m(float(sa["stop_lat"]), float(sa["stop_lon"]),
                             float(sb["stop_lat"]), float(sb["stop_lon"]))
        return d

    def station(self, stop_id):
        """Stop place of a stop: GTFS 'parent_station' if given (platforms of one place)."""
        p = self.stops[stop_id].get("parent_station") or ""
        return p if p else stop_id

    def pattern(self, trip_id):
        return tuple(self.station(s["stop_id"]) for s in self.st[trip_id])


def main_patterns(w, line, date):
    """The most frequent stop pattern of each direction on the date, with one example trip."""
    by_dir = defaultdict(Counter)
    example = {}
    for t in w.line_trips(line, date):
        p = w.pattern(t["trip_id"])
        by_dir[t["direction_id"]][p] += 1
        example.setdefault((t["direction_id"], p), t["trip_id"])
    out = []
    for d in sorted(by_dir):
        p, n = by_dir[d].most_common(1)[0]
        out.append((d, p, n, example[(d, p)]))
    return out


# ---------------------------------------------------------------- road geometry from OpenStreetMap

def seg_project(p, a, b):
    """Distance from point p to segment ab and the position t in [0, 1] of the nearest point."""
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / L2))
    qx, qy = ax + t * dx, ay + t * dy
    return math.hypot(p[0] - qx, p[1] - qy), t


def nearest_node(nodes, p):
    return min(range(len(nodes)), key=lambda i: math.dist(nodes[i], p))


def sub_path(nodes, i, j):
    """Nodes of one way from index i to index j. A closed way (a roundabout, drawn in the
    driving direction) is followed forward and wraps around; an open way is reversed when
    i > j."""
    closed = len(nodes) > 2 and math.dist(nodes[0], nodes[-1]) < 0.5
    if i <= j:
        return nodes[i:j + 1]
    if closed:
        return nodes[i:-1] + nodes[0:j + 1]
    return nodes[j:i + 1][::-1]


def chain_ways(rel):
    """Join the ways of an OSM route relation, in member order, into one polyline (LV95).
    A route often uses only part of a way (a roundabout, a long street), so each way is cut:
    it is entered at its node nearest to the end of the polyline so far, and left at its node
    nearest to the next way. Returns the polyline and the largest gap (m) between two
    consecutive ways (a large gap means a broken relation)."""
    ways = [m for m in rel["members"] if m["type"] == "way" and "geometry" in m
            and m.get("role", "") in ("", "forward", "backward")]
    segs = [[wgs84_to_lv95(g["lat"], g["lon"]) for g in m["geometry"]] for m in ways]
    if not segs:
        return [], float("inf")
    if len(segs) == 1:
        return segs[0], 0.0

    def exit_index(k):
        """Node of way k nearest to way k + 1."""
        nxt = segs[k + 1]
        return min(range(len(segs[k])),
                   key=lambda i: min(math.dist(segs[k][i], q) for q in nxt))

    # First way: leave at the node nearest to way 2, enter at the end farther from it.
    j = exit_index(0)
    s0 = segs[0]
    i = 0 if j > (len(s0) - 1) / 2 else len(s0) - 1
    line = list(sub_path(s0, i, j))
    gap = 0.0
    for k in range(1, len(segs)):
        s = segs[k]
        i = nearest_node(s, line[-1])
        if k < len(segs) - 1:
            j = exit_index(k)
        else:
            j = len(s) - 1 if i < (len(s) - 1) / 2 else 0
        part = sub_path(s, i, j)
        gap = max(gap, math.dist(line[-1], part[0]))
        line.extend(part[1:] if math.dist(line[-1], part[0]) < 0.5 else part)
    return line, gap


def locate(poly, cum, p):
    """Nearest point of the polyline to p: (distance to p, position along the polyline in m)."""
    best = (float("inf"), 0.0)
    for i in range(len(poly) - 1):
        d, t = seg_project(p, poly[i], poly[i + 1])
        if d < best[0]:
            best = (d, cum[i] + t * (cum[i + 1] - cum[i]))
    return best


def cut(poly, cum, s0, s1):
    """Part of the polyline between the positions s0 < s1 (m)."""
    def at(s):
        for i in range(len(poly) - 1):
            if cum[i] <= s <= cum[i + 1]:
                f = 0.0 if cum[i + 1] == cum[i] else (s - cum[i]) / (cum[i + 1] - cum[i])
                return (poly[i][0] + f * (poly[i + 1][0] - poly[i][0]),
                        poly[i][1] + f * (poly[i + 1][1] - poly[i][1]))
        return poly[-1]
    inner = [poly[i] for i in range(len(poly)) if s0 < cum[i] < s1]
    return [at(s0)] + inner + [at(s1)]


def simplify(poly, tol):
    """Douglas-Peucker simplification with tolerance tol (m)."""
    if len(poly) < 3:
        return poly
    keep = [False] * len(poly)
    keep[0] = keep[-1] = True
    stack = [(0, len(poly) - 1)]
    while stack:
        i, j = stack.pop()
        dmax, k = 0.0, -1
        for m in range(i + 1, j):
            d, _ = seg_project(poly[m], poly[i], poly[j])
            if d > dmax:
                dmax, k = d, m
        if dmax > tol and k > 0:
            keep[k] = True
            stack += [(i, k), (k, j)]
    return [p for p, f in zip(poly, keep) if f]


def cumlen(poly):
    c = [0.0]
    for a, b in zip(poly, poly[1:]):
        c.append(c[-1] + math.dist(a, b))
    return c


def osm_road(osm, line, stops_lv95, max_offset=50.0, max_end_offset=150.0):
    """Road geometry of one stop pattern from the OSM relations with the same line number
    (tag ref) in the ZVV network. A relation fits when it passes all the stops in their order,
    within max_offset metres (intermediate stops) and max_end_offset metres (the two termini).
    Among the relations that fit, the one with the shortest road from the first to the last
    stop is chosen: a longer one makes a detour that this stop pattern does not serve. The
    polyline is cut at the first and the last stop. Returns a dict with the relation id, the
    road length, the largest stop offsets and the largest gap between ways, or None."""
    best = None
    for rel in osm:
        t = rel.get("tags", {})
        if t.get("ref") != line or t.get("network") != "ZVV":
            continue
        poly, gap = chain_ways(rel)
        if len(poly) < 2:
            continue
        cum = cumlen(poly)
        loc = [locate(poly, cum, p) for p in stops_lv95]
        # Offsets of the intermediate stops; a terminus is often a station centroid away from
        # the road, so its offset is reported apart (max_end_offset_m).
        off = max([d for d, _ in loc[1:-1]] or [0.0])
        end_off = max(loc[0][0], loc[-1][0])
        order_ok = all(loc[i][1] <= loc[i + 1][1] + 1.0 for i in range(len(loc) - 1))
        s0, s1 = loc[0][1], loc[-1][1]
        fits = order_ok and off <= max_offset and end_off <= max_end_offset and s1 > s0
        # Relations that fit come first, the shortest of them wins; the others are ranked by
        # their offsets (kept only to report why no relation fits).
        score = (s1 - s0) if fits else 1e9 + off + end_off / 4.0
        if best is None or score < best["score"]:
            road = cut(poly, cum, s0, s1) if s1 > s0 else []
            best = {"score": score, "osm_relation": rel["id"], "road_m": s1 - s0,
                    "max_stop_offset_m": off, "max_end_offset_m": end_off,
                    "max_way_gap_m": gap, "order_ok": order_ok, "poly": road}
    return best


# ---------------------------------------------------------------- step 2: geometries

PROFILE_URL = "https://api3.geo.admin.ch/rest/services/profile.json"


def step_geoms(args):
    w = Work(args.work)
    osm = []
    if args.osm:
        with open(args.osm, encoding="utf-8", newline="") as f:
            osm = [e for e in json.load(f)["elements"] if e["type"] == "relation"]
    rows = []
    for line in args.lines.split(","):
        for d, p, n, tid in main_patterns(w, line, args.date):
            coords = []
            for s in w.st[tid]:
                stop = w.stops[s["stop_id"]]
                coords.append(wgs84_to_lv95(float(stop["stop_lat"]), float(stop["stop_lon"])))
            length = w.trip_length_m(tid)
            road = (osm_road(osm, line, coords, args.max_offset, args.max_end_offset)
                    if osm else None)
            # The road from OSM is used for the profile when it passes within args.max_offset
            # metres of every stop, in the right order; else the stop-to-stop polyline is used.
            use_road = bool(road and road["poly"] and road["order_ok"]
                            and road["max_stop_offset_m"] <= args.max_offset
                            and road["max_end_offset_m"] <= args.max_end_offset
                            and road["max_way_gap_m"] <= args.max_gap)
            geom_pts = road["poly"] if use_road else coords
            tol = 2.0
            while True:
                pts = simplify(geom_pts, tol)
                geom = json.dumps({"type": "LineString",
                                   "coordinates": [[round(x, 1), round(y, 1)] for x, y in pts]},
                                  separators=(",", ":"))
                if len(quote(geom)) < args.max_url:  # keep the GET request short
                    break
                tol *= 1.5
            glen = road["road_m"] if use_road else length
            # About one height sample every 10 m; the service limits the number of points.
            nb = max(50, min(args.max_points, int(glen / 10)))
            url = "%s?geom=%s&sr=2056&nb_points=%d&distinct_points=true" % (
                PROFILE_URL, quote(geom), nb)
            rows.append({"line": line, "direction_id": d, "trip_id": tid, "n_trips": n,
                         "n_stops": len(coords), "length_m": round(length),
                         "first_stop": w.stops[w.st[tid][0]["stop_id"]]["stop_name"],
                         "last_stop": w.stops[w.st[tid][-1]["stop_id"]]["stop_name"],
                         "osm_relation": road["osm_relation"] if road else "",
                         "osm_road_m": round(road["road_m"]) if road else "",
                         "osm_max_stop_offset_m": round(road["max_stop_offset_m"], 1) if road else "",
                         "osm_max_end_offset_m": round(road["max_end_offset_m"], 1) if road else "",
                         "osm_max_way_gap_m": round(road["max_way_gap_m"], 1) if road else "",
                         "osm_order_ok": road["order_ok"] if road else "",
                         "profile_on": "osm_road" if use_road else "stops",
                         "simplify_tol_m": round(tol, 1), "nb_points": nb, "url": url})
    write_csv(os.path.join(args.work, "patterns.csv"), rows, list(rows[0].keys()))
    for r in rows:
        print(r["line"], r["direction_id"], r["n_stops"], r["length_m"], r["osm_road_m"],
              r["osm_max_stop_offset_m"], r["osm_max_end_offset_m"], r["osm_max_way_gap_m"],
              r["osm_order_ok"],
              r["profile_on"], r["simplify_tol_m"], len(r["url"]), r["first_stop"], "->",
              r["last_stop"], "trips", r["n_trips"])


# ---------------------------------------------------------------- step 3: table


SMOOTH_STEP_M = 20.0     # resampling step of the height profile
SMOOTH_WINDOW = 5        # moving average over 5 samples = 100 m
GRADE_WINDOW_M = 100.0   # the steepest gradient is measured over 100 m of road


def profile_stats(path):
    """Total ascent, total descent (m) and steepest gradient (%) of a swisstopo profile.
    Heights: model COMB of the service (the terrain model, not the road deck: bridges and
    underpasses give false steps). To damp such steps and the noise of the model, the profile
    is resampled every SMOOTH_STEP_M metres and smoothed by a moving average over
    SMOOTH_WINDOW samples. The steepest gradient is the largest |height change| over
    GRADE_WINDOW_M of smoothed profile."""
    with open(path, encoding="utf-8", newline="") as f:
        pts = json.load(f)
    dist = [float(p["dist"]) for p in pts]
    alt = [float(p["alts"]["COMB"]) for p in pts]
    # Linear resampling at a fixed step.
    xs, zs = [], []
    x, k = 0.0, 0
    while x <= dist[-1]:
        while k < len(dist) - 2 and dist[k + 1] < x:
            k += 1
        d0, d1 = dist[k], dist[k + 1]
        f = 0.0 if d1 == d0 else (x - d0) / (d1 - d0)
        xs.append(x)
        zs.append(alt[k] + f * (alt[k + 1] - alt[k]))
        x += SMOOTH_STEP_M
    h = SMOOTH_WINDOW // 2
    sm = [sum(zs[max(0, i - h):i + h + 1]) / len(zs[max(0, i - h):i + h + 1])
          for i in range(len(zs))]
    up = sum(max(0.0, b - a) for a, b in zip(sm, sm[1:]))
    down = sum(max(0.0, a - b) for a, b in zip(sm, sm[1:]))
    n = int(round(GRADE_WINDOW_M / SMOOTH_STEP_M))
    steep = max([abs(sm[i + n] - sm[i]) / (xs[i + n] - xs[i]) * 100.0
                 for i in range(len(sm) - n)] or [0.0])
    return {"ascent_m": up, "descent_m": down, "max_grade_pct": steep,
            "z_start": alt[0], "z_end": alt[-1], "z_min": min(alt), "z_max": max(alt),
            "profile_len_m": dist[-1]}


def min_fleet(w, trips, layover_s, empty_kmh, detour):
    """Minimum number of vehicles that serve all the line's trips of one day.
    Trip j may follow trip i on the same vehicle when
        arrival_i + layover + empty_run_time(end_i, start_j) <= departure_j,
    with empty_run_time = detour * straight distance / empty_kmh (zero at the same stop place).
    The minimum number of vehicles is (number of trips) - (largest matching of 'i followed by
    j' pairs): each matched pair saves one vehicle (minimum path cover of the trip graph).
    The matching is found with augmenting paths (Kuhn's algorithm)."""
    n = len(trips)
    if n == 0:
        return 0
    dep, arr, a_pt, b_pt = [], [], [], []
    for t in trips:
        st = w.st[t["trip_id"]]
        s0, s1 = w.stops[st[0]["stop_id"]], w.stops[st[-1]["stop_id"]]
        dep.append(seconds(st[0]["departure_time"]))
        arr.append(seconds(st[-1]["arrival_time"]))
        a_pt.append((float(s0["stop_lat"]), float(s0["stop_lon"]), w.station(st[0]["stop_id"])))
        b_pt.append((float(s1["stop_lat"]), float(s1["stop_lon"]), w.station(st[-1]["stop_id"])))
    succ = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j or dep[j] < arr[i]:
                continue
            if b_pt[i][2] == a_pt[j][2]:
                empty = 0.0
            else:
                d = haversine_m(b_pt[i][0], b_pt[i][1], a_pt[j][0], a_pt[j][1])
                empty = detour * d / (empty_kmh / 3.6)
            if arr[i] + layover_s + empty <= dep[j]:
                succ[i].append(j)
    match_of = [-1] * n            # match_of[j] = trip i that j follows

    def augment(i, seen):
        # Iterative depth-first search for an augmenting path from trip i.
        stack = [(i, iter(succ[i]))]
        path = []
        while stack:
            u, it = stack[-1]
            moved = False
            for j in it:
                if j in seen:
                    continue
                seen.add(j)
                if match_of[j] == -1:
                    path.append((u, j))
                    for (uu, jj) in path:
                        match_of[jj] = uu
                    return True
                path.append((u, j))
                stack.append((match_of[j], iter(succ[match_of[j]])))
                moved = True
                break
            if not moved:
                stack.pop()
                if path:
                    path.pop()
        return False

    matched = 0
    for i in sorted(range(n), key=lambda k: arr[k]):
        if augment(i, set()):
            matched += 1
    return n - matched


def median(v):
    v = sorted(v)
    n = len(v)
    return None if n == 0 else (v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2.0)


def fmt_hm(s):
    return "%02d:%02d" % (s // 3600, (s % 3600) // 60)


def step_table(args):
    w = Work(args.work)
    prow = read_csv(os.path.join(args.work, "patterns.csv"))
    # The profiles are given in the order of the rows of patterns.csv.
    if len(args.profiles) != len(prow):
        sys.exit("give one profile file per row of patterns.csv (%d)" % len(prow))
    pats = {}
    for r, path in zip(prow, args.profiles):
        r["prof"] = profile_stats(path)
        pats[(r["line"], r["direction_id"])] = r
    lay = args.layover_min * 60
    out = []
    for line in args.lines.split(","):
        dirs = sorted(d for (l, d) in pats if l == line)
        P = [pats[(line, d)] for d in dirs]
        row = {"line": line,
               "terminus_a": P[0]["first_stop"], "terminus_b": P[0]["last_stop"]}
        # Route length per direction: the OSM road when it was accepted, else the
        # stop-to-stop straight lines (a lower bound).
        factor = {}
        main_pat = {}
        L, T = [], []
        for i, p in enumerate(P):
            on_road = p["profile_on"] == "osm_road"
            length = float(p["osm_road_m"]) if on_road else float(p["length_m"])
            factor[p["direction_id"]] = length / float(p["length_m"])
            main_pat[p["direction_id"]] = w.pattern(p["trip_id"])
            # Running time: median over the weekday trips of the main pattern.
            runs = [(seconds(w.st[t["trip_id"]][-1]["arrival_time"])
                     - seconds(w.st[t["trip_id"]][0]["departure_time"])) / 60.0
                    for t in w.line_trips(line, args.weekday)
                    if t["direction_id"] == p["direction_id"]
                    and w.pattern(t["trip_id"]) == main_pat[p["direction_id"]]]
            run = median(runs)
            L.append(length)
            T.append(run)
            pr = p["prof"]
            sfx = "_d%d" % i
            row["length_km" + sfx] = round(length / 1000.0, 2)
            row["length_source" + sfx] = "OSM road" if on_road else "stops, straight lines"
            row["length_stops_km" + sfx] = round(float(p["length_m"]) / 1000.0, 2)
            row["n_stops" + sfx] = int(p["n_stops"])
            row["run_time_min" + sfx] = round(run, 1)
            row["ascent_m" + sfx] = round(pr["ascent_m"])
            row["descent_m" + sfx] = round(pr["descent_m"])
            row["max_grade_pct" + sfx] = round(pr["max_grade_pct"], 1)
        nst = sum(int(p["n_stops"]) for p in P)
        row["stops_per_km"] = round(nst / (sum(L) / 1000.0), 2)
        row["mean_speed_kmh"] = round((sum(L) / 1000.0) / (sum(T) / 60.0), 1)
        row["z_min_m"] = round(min(p["prof"]["z_min"] for p in P))
        row["z_max_m"] = round(max(p["prof"]["z_max"] for p in P))
        for label, date in (("wd", args.weekday), ("sa", args.saturday), ("su", args.sunday)):
            trips = w.line_trips(line, date)
            # Vehicle-km: each trip's stop-to-stop length times the road factor of its
            # direction (road length / stop-to-stop length of the main pattern).
            km = sum(w.trip_length_m(t["trip_id"]) * factor.get(t["direction_id"], 1.0)
                     for t in trips) / 1000.0
            n = min_fleet(w, trips, lay, args.empty_kmh, args.detour) if trips else 0
            deps = [seconds(w.st[t["trip_id"]][0]["departure_time"]) for t in trips]
            row["trips_" + label] = len(trips)
            row["n_j_" + label] = n
            row["veh_km_" + label] = round(km, 1)
            row["km_per_vehicle_" + label] = round(km / n, 1) if n else ""
            row["first_dep_" + label] = fmt_hm(min(deps)) if deps else ""
            row["last_dep_" + label] = fmt_hm(max(deps)) if deps else ""
        # Cross-check of n_j on the weekday: cycle time / headway in the hour 07:00-08:00,
        # n = ceil((T_d0 + T_d1 + 2 * layover) / headway), headway = 60 / departures of
        # direction 0 in that hour.
        d0 = [t for t in w.line_trips(line, args.weekday) if t["direction_id"] == P[0]["direction_id"]]
        k = sum(1 for t in d0 if 7 * 3600 <= seconds(w.st[t["trip_id"]][0]["departure_time"]) < 8 * 3600)
        if k and len(T) == 2:
            hw = 60.0 / k
            row["headway_min_7to8_wd"] = round(hw, 1)
            row["n_cycle_wd"] = math.ceil((T[0] + T[1] + 2 * args.layover_min) / hw)
        else:
            row["headway_min_7to8_wd"] = ""
            row["n_cycle_wd"] = ""
        out.append(row)
    fields = []
    for r in out:
        for f in r:
            if f not in fields:
                fields.append(f)
    write_csv(args.out, out, fields)
    print("wrote", args.out, len(out), "rows; feed version", w.feed.get("feed_version"))




# ---------------------------------------------------------------- command line

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("filter")
    a.add_argument("gtfs")
    a.add_argument("work")
    a.add_argument("--agency", required=True)
    a.set_defaults(func=step_filter)
    b = sub.add_parser("geoms")
    b.add_argument("work")
    b.add_argument("--lines", required=True)
    b.add_argument("--date", required=True)
    b.add_argument("--max-points", type=int, default=1000,
                   help="largest number of height samples per profile")
    b.add_argument("--max-url", type=int, default=3800,
                   help="largest length of the encoded geometry in the request URL (the service refuses request lines over 4094 characters)")
    b.add_argument("--osm", help="Overpass JSON with the OSM bus route relations (out geom)")
    b.add_argument("--max-offset", type=float, default=50.0,
                   help="largest distance (m) of an intermediate stop from the OSM road")
    b.add_argument("--max-end-offset", type=float, default=150.0,
                   help="largest distance (m) of the first or last stop from the OSM road")
    b.add_argument("--max-gap", type=float, default=10.0,
                   help="largest gap (m) between two consecutive OSM ways")
    b.set_defaults(func=step_geoms)
    c = sub.add_parser("table")
    c.add_argument("work")
    c.add_argument("profiles", nargs="+")
    c.add_argument("--lines", required=True)
    c.add_argument("--weekday", required=True)
    c.add_argument("--saturday", required=True)
    c.add_argument("--sunday", required=True)
    c.add_argument("--layover-min", type=float, default=3.0,
                   help="minimum layover (min) between two trips of one vehicle")
    c.add_argument("--empty-kmh", type=float, default=20.0,
                   help="speed (km/h) of an empty run between two stop places")
    c.add_argument("--detour", type=float, default=1.3,
                   help="road length / straight distance of an empty run")
    c.add_argument("--out", required=True)
    c.set_defaults(func=step_table)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    args.func(args)


if __name__ == "__main__":
    main()
