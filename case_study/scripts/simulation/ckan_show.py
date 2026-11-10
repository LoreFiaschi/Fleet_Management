"""Print the main fields and the resources of a CKAN package_show JSON file.
Usage: python -I ckan_show.py <file.json>"""
import json, sys, re
d = json.load(open(sys.argv[1], encoding="utf-8"))["result"]
for k in ("name", "title", "license_title", "metadata_modified", "dateFirstPublished", "updateInterval", "timeRange", "author", "maintainer"):
    print(k, ":", d.get(k))
notes = re.sub(r"\s+", " ", d.get("notes") or "")
print("notes:", notes[:3000])
for r in d.get("resources", []):
    print("RES |", r.get("name"), "|", r.get("format"), "|", r.get("size"), "|", r.get("url"))
