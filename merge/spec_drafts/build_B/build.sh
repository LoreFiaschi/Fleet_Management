cd /c/Users/utente/Documents/GitHub/Fleet_Management/merge/spec_drafts/
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build_B wrap_B.tex > /dev/null 2>&1
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build_B wrap_B.tex > build_B/run.txt 2>&1
echo exit=$?
grep -n "^!\|Overfull\|Underfull\|Float too large" build_B/wrap_B.log | head -20
echo "undefined refs (other writers expected):"
grep "Reference" build_B/wrap_B.log | sed 's/.*Reference .\([a-z_:]*\).*/\1/' | sort -u | tr '\n' ' '; echo
grep "Citation" build_B/wrap_B.log | head
grep "Output written" build_B/wrap_B.log
