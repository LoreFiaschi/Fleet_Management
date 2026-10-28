"""Assemble spec version 1.0 from the writers' fragments (Claude, 26 Oct 2026).

Run from merge/spec_drafts/: python -I assemble.py
Writes spec_assembled.tex in this folder. spec/spec.tex is replaced only by hand, after the checks.
The spec holds only the specification: the changelog and the open questions are in scoping.tex.
"""
import re

TITLE = "Software Design Document"   # the document type on the title page

def read(p):
    with open(p, encoding="utf-8", newline="") as f:
        return f.read().replace("\r\n", "\n")

def blocks(text):
    """Split a fragment at its '% ==== BLOCK n ====' lines."""
    parts = re.split(r"^% ==== BLOCK [0-9A-Z]+ ====\s*$", text, flags=re.M)
    return [p.strip("\n") + "\n" for p in parts]

def split_before_section(text, title):
    """Split text before the separator lines that precede \\section{title}."""
    lines = text.split("\n")
    idx = next(i for i, l in enumerate(lines) if l.startswith("\\section{" + title))
    while idx > 0 and lines[idx - 1].startswith("%"):
        idx -= 1
    return "\n".join(lines[:idx]).rstrip("\n") + "\n\n", "\n".join(lines[idx:])

preamble = read("preamble.tex")

title = r"""\begin{document}

% --- Title page ---
\hypersetup{pageanchor=false}
\begin{titlepage}
  \centering
  \vspace*{3cm}
  {\Huge\bfseries Fleet Management\par}
  \vspace{0.5cm}
  {\Large """ + TITLE + r"""\par}
  \vspace{0.3cm}
  {\large Multi-Component Stochastic Degradation MILP\par}
  \vspace{1cm}
  {\large Version 1.0 $\cdot$ \today\par}
\end{titlepage}
\hypersetup{pageanchor=true}

\tableofcontents
\newpage

"""

A = blocks(read("fragments_A.tex"))
B = blocks(read("fragments_B.tex"))
C = blocks(read("fragments_C.tex"))
D = blocks(read("fragments_D.tex"))
assert len(A) == 3 and len(B) == 1 and len(C) == 3 and len(D) in (1, 2), (len(A), len(B), len(C), len(D))
overview, notation = split_before_section(A[0], "Index Sets and Notation")
if len(D) == 2:
    architecture, tools = D[0], D[1]
else:
    architecture, tools = "", D[0]

# writer C's table macros go into the preamble
c_lines = C[0].split("\n")
macro_lines = [l for l in c_lines[:12] if l.startswith("\\renewcommand{\\_}") or l.startswith("\\providecommand{\\")]
assert len(macro_lines) >= 5, macro_lines
C[0] = "\n".join(l for l in c_lines if l not in macro_lines).lstrip("\n")
preamble = preamble.rstrip("\n") + "\n\n% --- Table rows (input and output tables) ---\n" + "\n".join(macro_lines) + "\n"

parts = [preamble, "\n", title, overview, architecture, "\n", notation, "\n", C[0], "\n", A[1], "\n",
         B[0], "\n", A[2], "\n", read("claude_full_problem.tex"), "\n", C[1], "\n", tools, "\n", C[2],
         "\n\\end{document}\n"]
out = "".join(parts)
out = re.sub(r"\n{4,}", "\n\n\n", out)
with open("spec_assembled.tex", "w", encoding="utf-8", newline="\n") as f:
    f.write(out)
print("lines:", out.count("\n"))
