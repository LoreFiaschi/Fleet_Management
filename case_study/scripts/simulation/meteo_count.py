"""Count hourly air temperature (tre200h0) in MeteoSwiss SMN 'h_recent' files.
Usage: python -I meteo_count.py <csv> [...]"""
import csv, sys, statistics
for p in sys.argv[1:]:
    with open(p, encoding="cp1252", newline="") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    t = [float(r["tre200h0"]) for r in rows if r["tre200h0"] not in ("", "-")]
    print(rows[0]["station_abbr"], "rows", len(rows), "first", rows[0]["reference_timestamp"], "last", rows[-1]["reference_timestamp"],
          "tre200h0 values", len(t), "min %.1f mean %.1f max %.1f" % (min(t), statistics.mean(t), max(t)))
