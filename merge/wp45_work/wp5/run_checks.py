"""Run the four local checks of the plan's section 3 in one go.

Usage: python run_checks.py
Parses base.py with ast, checks the line endings (all CRLF), then runs
load_base.py, names_check.py and local_checks.py and prints their output.
"""
import ast
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
BASE = pathlib.Path(
    r"C:\Users\utente\Documents\GitHub\Fleet_Management-merge"
    r"\src\fleet_management\degradation_model\base.py"
)

data = BASE.read_bytes()
ast.parse(data.decode("utf-8"))
print("ast.parse OK; lines", data.count(b"\n"), "; CRLF", data.count(b"\r\n"), "; lone LF", data.count(b"\n") - data.count(b"\r\n"))
for script in ("load_base.py", "names_check.py", "local_checks.py"):
    print(f"--- {script}")
    result = subprocess.run([sys.executable, script], cwd=HERE, capture_output=True, text=True, encoding="utf-8")
    out = result.stdout + result.stderr
    if script == "load_base.py":
        print("exit", result.returncode, "|", out.strip()[:400] if result.returncode else "loaded")
    elif script == "local_checks.py":
        print("\n".join(line for line in out.splitlines() if not line.startswith("PASS ")))
    else:
        print(out.strip())
