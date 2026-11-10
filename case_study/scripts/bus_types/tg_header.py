"""Print the column headers of the ASTRA type-approval file with their indices.
Usage: python -E tg_header.py TG-Automobil.txt"""
import sys

with open(sys.argv[1], "r", encoding="cp1252", errors="replace", newline="") as f:
    head = f.readline().rstrip("\r\n").split("\t")
for i, h in enumerate(head):
    print(i, repr(h))
