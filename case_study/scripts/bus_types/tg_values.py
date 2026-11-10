"""Count the values of one column of the ASTRA file. Usage: python -E tg_values.py file col [col2]"""
import sys
import collections

cols = [int(x) for x in sys.argv[2:]]
cnt = collections.Counter()
lens = collections.Counter()
with open(sys.argv[1], "r", encoding="cp1252", errors="replace", newline="") as f:
    f.readline()
    for line in f:
        c = line.rstrip("\r\n").split("\t")
        lens[len(c)] += 1
        cnt[tuple(c[i] if i < len(c) else "" for i in cols)] += 1
print("field counts per row:", lens.most_common(5))
for k, v in cnt.most_common(60):
    print(v, k)
