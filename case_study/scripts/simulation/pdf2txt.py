"""Extract the text of a PDF with pypdf, one page marker per page.
Usage: python -I pdf2txt.py <in.pdf> <out.txt>"""
import sys
from pypdf import PdfReader

r = PdfReader(sys.argv[1])
parts = []
for i, pg in enumerate(r.pages, 1):
    try:
        t = pg.extract_text() or ""
    except Exception as e:  # keep going on a bad page
        t = "[extraction error: %s]" % e
    parts.append("\n===== PAGE %d =====\n%s" % (i, t))
res = "".join(parts)
open(sys.argv[2], "w", encoding="utf-8", newline="\n").write(res)
print(len(r.pages), "pages,", len(res), "chars")
