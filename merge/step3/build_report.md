Status: done (8 Nov 2026). Both documents built at 0 / 0 / 0 with no overfull box and no "Float too large".

# Task 1: edits of `merge/spec_drafts/fragments_C.tex`
- `completion` key `status`: added `"error"` (Gurobi raised an error).
- `relaxation_warm_start`: new row "`status` is `"error"`" (Gurobi raised an error; `relaxation_objective` null, `values_copied` 0).
- `performance` table: `model_construction_seconds` is "Model construction, including all warm-start phases (mapping, completion, relaxation start)".

# Task 2: `scoping.tex`, section `sec:acceptance`
- The two history sentences are replaced by the two sentences given (no unsafe schedule accepted; loop rows from day T to day H1 refuse 1512 safe schedules).
- "The anchor is no longer fixed on day H1 ..." deleted.
- Tell-word search (corrected, earlier, was described as, is wrong, no longer, now, previously, before the change, new rows, introduced) over `scoping.tex` and all fragments: no hit in text added by the writers. The hits that remain are old text or the change-log appendix of the scoping document (lines 3542, 3570, 4134, 4167-4209) and "introduced the Gamma process" (line 915); none was changed. No tell-word fix was needed.

# Task 3: build
- No latexmk, pdflatex or bibtex ran. `assemble.py` gave 3009 lines. The diff of `spec_assembled.tex` against `spec/spec.tex` showed only the writers' changes (package layout, data flow, cell-builder interface, ...). Copied over `spec/spec.tex` with CRLF.
- No undefined reference, so no label fix was needed.
- Build fix (first spec build had 3 overfull boxes, in the Cell-Builder Interface of `fragments_D.tex`): the call signatures of `add_quantity_rows` and of the three reading helpers are now written as name plus "arguments ..." prose; the first column of the table "Errors of the builders" is 4.2 cm wide (was 3.4 cm).

| Document | Pages | Errors | Undefined | bibtex warnings | Overfull | Float too large |
|---|---|---|---|---|---|---|
| `spec/spec.pdf` | 62 | 0 | 0 | 0 | 0 | 0 |
| `scoping.pdf` | 92 | 0 | 0 | 0 | 0 | 0 |

# Task 4: pages read (70 dpi, pdftocairo; PDF page numbers)
- Spec: 5, 6 (package layout, data flow, builder interface), 7, 8 (context tables, helpers, builder errors, building order), 16, 17 (decision variables table starts on 17, validator lists), 30 (damage penalty), 31 (Loop Constraint), 32 (Result Dictionary), 49, 50 (Structural Validator tables).
- Scoping: 10, 11 (acceptance section, the new sentences), 74 (proof of perpetual safety).
- All tables fit the page width; nothing overlaps.
