"""Copy a UTF-8 text file and write it with CRLF line endings (WP6 Gamma plan, steps 3 and 5).

Usage: python copy_crlf.py <source> <destination>
"""
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_bytes().decode("utf-8")
text = text.replace("\r\n", "\n").replace("\n", "\r\n")
Path(sys.argv[2]).write_bytes(text.encode("utf-8"))
print(f"wrote {sys.argv[2]}: {text.count(chr(10))} lines, CRLF")
