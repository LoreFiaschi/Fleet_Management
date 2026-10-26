"""Assemble spec version 1.0 from the writers' fragments (Claude, 26 Oct 2026).

Run from merge/spec_drafts/: python -I assemble.py
Writes spec_assembled.tex in this folder. spec/spec.tex is replaced only by hand, after the checks.
"""
import re

ROOT = "../../"

def read(p):
    with open(p, encoding="utf-8", newline="") as f:
        return f.read().replace("\r\n", "\n")

def blocks(text):
    """Split a fragment at its '% ==== BLOCK n ====' lines."""
    parts = re.split(r"^% ==== BLOCK \d+ ====\s*$", text, flags=re.M)
    return [p.strip("\n") + "\n" for p in parts]

def split_before_section(text, title):
    """Split text before the separator lines that precede \\section{title}."""
    lines = text.split("\n")
    idx = next(i for i, l in enumerate(lines) if l.startswith("\\section{" + title))
    while idx > 0 and lines[idx - 1].startswith("%"):
        idx -= 1
    return "\n".join(lines[:idx]).rstrip("\n") + "\n\n", "\n".join(lines[idx:])

old = read(ROOT + "spec/spec.tex")
preamble = read("preamble.tex")

title = r"""\begin{document}

% --- Title page ---
\begin{titlepage}
  \centering
  \vspace*{3cm}
  {\Huge\bfseries Fleet Management\par}
  \vspace{0.5cm}
  {\Large Software Requirements Specification\par}
  \vspace{0.3cm}
  {\large Multi-Component Stochastic Degradation MILP\par}
  \vspace{1cm}
  {\large Version 1.0 $\cdot$ \today\par}
  \vfill
  {\small Prepared with spec-creator $\cdot$ Intended for Claude Code implementation}
\end{titlepage}

\tableofcontents
\newpage

"""

A = re.split(r"^% ==== BLOCK [0-9A-Z]+ ====\s*$", read("fragments_A.tex"), flags=re.M)
A = [p.strip("\n") + "\n" for p in A]
B = blocks(read("fragments_B.tex"))
C = blocks(read("fragments_C.tex"))
D = blocks(read("fragments_D.tex"))
assert len(A) == 3 and len(B) == 1 and len(C) == 3 and len(D) == 1, (len(A), len(B), len(C), len(D))
A1_head, A1_tail = A[0], A[1]
A = [None, A[2]]

# writer C's table-row macros go into the preamble
c_lines = C[0].split("\n")
macro_lines = [l for l in c_lines[:8] if l.startswith("\\renewcommand{\\_}") or l.startswith("\\providecommand{\\")]
assert len(macro_lines) == 5, macro_lines
C[0] = "\n".join(l for l in c_lines if l not in macro_lines).lstrip("\n")
preamble = preamble.rstrip("\n") + "\n\n% --- Table rows (input and output tables) ---\n" + "\n".join(macro_lines) + "\n"

# old change log paragraphs, without the change markers
start = old.index("\\emph{Change log (v0.3")
end = old.index("\\end{document}")
old_log = "\n".join(l for l in old[start:end].split("\n") if not l.startswith("%% --- CHANGE")).strip("\n")
# the old log cites an equation of version 0.4 that version 1.0 states in another form
old_log = old_log.replace("Eq.~\\eqref{eq:rf_loop}", "the loop row")
changelog = ("% ================================================================\n\\section{Changelog}\n"
             "% ================================================================\n\n" + old_log + "\n\n"
             + read("claude_changelog_v10.tex"))

parts = [preamble, "\n", title, A1_head, C[0], "\n", A1_tail, "\n", B[0], "\n", A[1], "\n",
         read("claude_full_problem.tex"), "\n", C[1], "\n", D[0], "\n", C[2], "\n",
         read("claude_open.tex"), "\n", changelog, "\n", read("references.tex"), "\n\\end{document}\n"]
out = "".join(parts)
out = re.sub(r"\n{4,}", "\n\n\n", out)
with open("spec_assembled.tex", "w", encoding="utf-8", newline="\n") as f:
    f.write(out)
print("lines:", out.count("\n"))
