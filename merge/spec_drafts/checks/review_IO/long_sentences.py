import re
import sys

lines = open(sys.argv[1], encoding="utf-8").read().split("\n")
rng = list(range(258, 686)) + list(range(2883, 4308))
text = " ".join(
    "¶%d¶ %s" % (n, lines[n - 1])
    for n in rng
    if not lines[n - 1].lstrip().startswith("%")
)
sents = re.split(r"(?<=[a-z0-9\)\}\$])\.\s+(?=[A-Z\\¶])", text)
res = []
for s in sents:
    nums = re.findall("¶(\\d+)¶", s)
    clean = re.sub("¶\\d+¶", " ", s)
    clean = re.sub(r"\$[^$]*\$", " X ", clean)
    clean = re.sub(r"\\[a-zA-Z]+", " ", clean)
    w = len(re.findall(r"[A-Za-z0-9_]+", clean))
    res.append((w, nums[0] if nums else "?", clean.strip()[:140]))
res.sort(reverse=True)
for w, n, c in res[:12]:
    print(w, n, c)
