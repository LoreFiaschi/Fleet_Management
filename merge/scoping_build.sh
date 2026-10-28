#!/bin/sh
# Usage: build.sh <outdir>   (builds scoping.tex from the repository into <outdir>)
REPO=/c/Users/utente/Documents/GitHub/Fleet_Management
OUT="$1"
mkdir -p "$OUT"
cd "$REPO" || exit 1
export TEXINPUTS="C:/Users/utente/Documents/GitHub/Fleet_Management;"
export BIBINPUTS="C:/Users/utente/Documents/GitHub/Fleet_Management;"
pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$OUT" scoping.tex > "$OUT/pass1.txt" 2>&1
echo "pass1 exit $?"
( cd "$OUT" && bibtex scoping > "$OUT/bibtex.txt" 2>&1 ; echo "bibtex exit $?" )
pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$OUT" scoping.tex > "$OUT/pass2.txt" 2>&1
echo "pass2 exit $?"
pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$OUT" scoping.tex > "$OUT/pass3.txt" 2>&1
echo "pass3 exit $?"
LOG="$OUT/scoping.log"
echo "errors (lines starting with !):"; grep -c '^!' "$LOG"
echo "undefined refs/cites:"; grep -c -i 'undefined' "$LOG"
grep -i 'undefined' "$LOG" | head -20
echo "float too large:"; grep -c 'Float too large' "$LOG"
echo "overfull:"; grep -c 'Overfull' "$LOG"
echo "underfull:"; grep -c 'Underfull' "$LOG"
echo "latex warnings:"; grep -c 'LaTeX Warning' "$LOG"
grep 'LaTeX Warning' "$LOG" | sort | uniq -c | head -20
echo "bibtex warnings/errors:"; grep -i -c 'warning\|error' "$OUT/scoping.blg"
grep -i 'warning\|error' "$OUT/scoping.blg" | head -20
pdfinfo "$OUT/scoping.pdf" 2>/dev/null | grep Pages
