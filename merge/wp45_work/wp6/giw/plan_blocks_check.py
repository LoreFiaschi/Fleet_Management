"""Compare the code blocks of merge/wp6_plan_giw.md with the prototypes. Usage: python plan_blocks_check.py"""
import re
from pathlib import Path
import config_patch as cp

plan = Path(r"C:\Users\utente\Documents\GitHub\Fleet_Management\merge\wp6_plan_giw.md").read_text(encoding="utf-8")
blocks = re.findall(r"```python\n(.*?)```", plan, flags=re.S)
print("code blocks:", len(blocks))
proto_g = Path("proto/degradation_model/gaussian.py").read_text(encoding="utf-8")
print("G1 block == proto gaussian.py:", blocks[0] == proto_g)
print("C2 block == config_patch SECTION:", blocks[1] == cp.SECTION.rstrip("\n") + "\n")
print("C3 block == config_patch IG check:", blocks[2] == cp.MARKOV_NEW[len(cp.MARKOV_OLD):])
print("C4 block == config_patch NEW_TESTS:", blocks[3] == cp.NEW_TESTS.lstrip("\n"))
